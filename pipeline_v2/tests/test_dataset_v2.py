# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pytest

import dataset_v2 as dv


def _base_sintetica(n_dias=1500, semente=7):
    rng = np.random.default_rng(semente)
    datas = pd.date_range("1990-01-01", periods=n_dias, freq="D")
    precip = rng.gamma(0.6, 6.0, n_dias)
    base = pd.DataFrame({
        "data": datas,
        "era5_precip_mm": precip,
        "era5_et0_mm": 3 + 2 * rng.random(n_dias),
        "era5_tmean_c": 20 + 5 * rng.standard_normal(n_dias),
        "power_precip_mm": precip * (0.9 + 0.2 * rng.random(n_dias)),
        "power_tmean_c": 20 + 5 * rng.standard_normal(n_dias),
        "vol_util_pct_sistema": np.clip(60 + np.cumsum(rng.standard_normal(n_dias)) * 0.3, -20, 100),
        "afluencia_m3s_sistema": 10 + 5 * rng.random(n_dias),
        "defluencia_m3s_sistema": 8 + 3 * rng.random(n_dias),
    })
    return base


# ---------------------------------------------------------------- causalidade
@pytest.mark.parametrize("corte", ["1992-06-15", "1993-03-01", "1993-12-31"])
def test_features_sao_causais(corte):
    """As features no dia T não podem mudar se a série for cortada em T
    (dependem só de t e do passado)."""
    base = _base_sintetica()
    T = pd.Timestamp(corte)

    feat_cheio = dv.construir_features(base)
    feat_corte = dv.construir_features(base[base["data"] <= T].copy())

    lin_cheio = feat_cheio.loc[feat_cheio["data"] == T].reset_index(drop=True)
    lin_corte = feat_corte.loc[feat_corte["data"] == T].reset_index(drop=True)

    assert len(lin_cheio) == 1 and len(lin_corte) == 1
    for col in feat_cheio.columns:
        if col == "data":
            continue
        a = lin_cheio[col].iloc[0]
        b = lin_corte[col].iloc[0]
        if pd.isna(a) or pd.isna(b):
            assert pd.isna(a) and pd.isna(b), f"{col}: NaN só de um lado em T={corte}"
        else:
            assert np.isclose(a, b, rtol=1e-9, atol=1e-9), \
                f"{col} diferente em T={corte}: cheio={a} corte={b}"


# ------------------------------------------------------------------ calendário
def test_calendario_sem_duplicatas_e_continuo():
    base = _base_sintetica(n_dias=900)
    # remove alguns dias das fontes p/ simular buracos; o calendário deve seguir contínuo
    sistema = base[["data", "vol_util_pct_sistema", "afluencia_m3s_sistema",
                    "defluencia_m3s_sistema"]].drop(index=range(100, 130))
    era5 = base[["data", "era5_precip_mm", "era5_et0_mm", "era5_tmean_c"]]
    power = base[["data", "power_precip_mm", "power_tmean_c"]]

    out, buracos = dv.construir_calendario(sistema, era5, power, ano_inicial=1990)
    assert out["data"].is_unique, "há datas duplicadas no calendário"
    difs = out["data"].diff().dropna().dt.days.unique()
    assert set(difs) == {1}, f"calendário não é diário contínuo: passos={difs}"
    # os 30 dias removidos do SAR devem aparecer como buracos
    linha_sar = buracos.set_index("fonte").loc["sar_sistema"]
    assert linha_sar["n_dias_ausentes"] >= 30


# ---------------------------------------------------------------------- alvos
def test_alvos_sao_o_volume_em_t_mais_h():
    base = _base_sintetica(n_dias=800)
    horizontes = [30, 60, 90]
    alvos = dv.construir_alvos(base, horizontes, limiar=30, limiar_sens=40)
    vol = base.set_index("data")["vol_util_pct_sistema"]

    for h in horizontes:
        s = alvos.set_index("data")[f"vol_pct_t{h}"]
        # para t cujo t+h existe, alvo(t) == vol(t+h)
        for t in [base["data"].iloc[0], base["data"].iloc[200], base["data"].iloc[-h - 1]]:
            esperado = vol.loc[t + pd.Timedelta(days=h)]
            assert np.isclose(s.loc[t], esperado, rtol=1e-9), \
                f"alvo t+{h} em {t.date()} != vol em t+{h}"
        # no fim da série (sem t+h) o alvo é NaN
        assert s.loc[base["data"].iloc[-1]] != s.loc[base["data"].iloc[-1]]  # NaN


def test_crise_binaria_coerente_com_limiar():
    base = _base_sintetica(n_dias=800)
    alvos = dv.construir_alvos(base, [30], limiar=30, limiar_sens=40)
    vol = base.set_index("data")["vol_util_pct_sistema"]
    j = alvos.set_index("data")
    t = base["data"].iloc[100]
    futuro = vol.loc[t + pd.Timedelta(days=30)]
    assert j.loc[t, "crise_30"] == float(futuro < 30)
    assert j.loc[t, "crise40_30"] == float(futuro < 40)


# ------------------------------------------------------------------ episódios
def test_detecta_episodio_e_respeita_duracao_minima():
    datas = pd.date_range("2000-01-01", periods=400, freq="D")
    vol = np.full(400, 70.0)
    vol[50:120] = 25.0          # episódio de 70 dias < 30%  -> deve entrar
    vol[200:210] = 10.0         # 10 dias < 30%  -> curto, não entra (dur_min=30)
    df = pd.DataFrame({"data": datas, "vol_util_pct_sistema": vol})
    epi = dv.detectar_episodios(df, "vol_util_pct_sistema", limiar=30,
                                gap_max_dias=60, dur_min_dias=30)
    assert len(epi) == 1
    assert epi.iloc[0]["duracao_dias"] == 70
    assert epi.iloc[0]["vol_min_pct"] == 25.0
