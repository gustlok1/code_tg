# -*- coding: utf-8 -*-
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import sintetico_v2 as sv

ROOT = Path(__file__).resolve().parents[2]
REAL = ROOT / "data" / "processed_v2" / "dataset_diario.parquet"
CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"

# climatologia simples (não depende do real) para testes de gerador
CLIM_FAKE = {"pocc": {m: 0.4 for m in range(1, 13)},
             "gshape": {m: 1.5 for m in range(1, 13)},
             "gscale": {m: 6.0 for m in range(1, 13)}}


def test_gerador_reprodutivel_por_seed():
    b1, c1 = sv.gerar_cenario(0.8, 10, seed=7, clim=CLIM_FAKE)
    b2, c2 = sv.gerar_cenario(0.8, 10, seed=7, clim=CLIM_FAKE)
    assert np.allclose(b1["era5_precip_mm"], b2["era5_precip_mm"])
    assert np.allclose(b1["vol_util_pct_sistema"], b2["vol_util_pct_sistema"])
    # seed diferente muda a série
    b3, _ = sv.gerar_cenario(0.8, 10, seed=8, clim=CLIM_FAKE)
    assert not np.allclose(b1["era5_precip_mm"], b3["era5_precip_mm"])


def test_conservacao_de_massa():
    _, comp = sv.gerar_cenario(0.5, 10, seed=1, clim=CLIM_FAKE)
    V = comp["V"].values
    dV = np.diff(V)
    bal = (comp["infl"] - comp["ret_eff"] - comp["evap"] - comp["vert"]).values[:-1]
    assert np.max(np.abs(dV - bal)) < 1e-6, "balanço hídrico não conserva massa"


def test_shuffle_deterministico_semente_do_config():
    # O modo shuffle da sanidade deriva a semente de shuffle_seed_base + índice do fold
    # (não de hash() de string, que é randomizado por processo). Duas chamadas no mesmo
    # processo têm de dar exatamente o mesmo skill (reprodução bit a bit).
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert "sintetico" in cfg and "shuffle_seed_base" in cfg["sintetico"], \
        "shuffle_seed_base ausente do config.yaml"
    base, _ = sv.gerar_cenario(0.8, 10, seed=0, clim=CLIM_FAKE)
    ds = sv.construir_dataset(base, cfg)
    s1, _ = sv.modelar_sintetico(ds, cfg, 90, modo="shuffle")
    s2, _ = sv.modelar_sintetico(ds, cfg, 90, modo="shuffle")
    assert s1 == s2, "o modo shuffle não é determinístico"
    # mudar a base da semente muda a permutação (logo, o skill)
    cfg2 = dict(cfg); cfg2["sintetico"] = {"shuffle_seed_base": cfg["sintetico"]["shuffle_seed_base"] + 1}
    s3, _ = sv.modelar_sintetico(ds, cfg2, 90, modo="shuffle")
    assert s3 != s1, "shuffle_seed_base não afeta o embaralhamento"


@pytest.mark.skipif(not REAL.exists(), reason="dataset real ainda não gerado")
def test_schema_identico_ao_real():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    base, _ = sv.gerar_cenario(0.0, 3, seed=0, clim=CLIM_FAKE)
    ds = sv.construir_dataset(base, cfg)
    real_cols = list(pd.read_parquet(REAL).columns)
    assert list(ds.columns) == real_cols, "schema do dataset sintético difere do real"


def test_todo_caminho_contem_sintetico():
    p = sv.caminho_cenario("phi0.8_N10_seed0")
    assert "sintetico" in str(p).lower()
    assert "sintetico" in str(sv.DIR_SINTETICO).lower()


@pytest.mark.skipif(not REAL.exists(), reason="dataset real ainda não gerado")
def test_phi_ef_mesma_funcao_no_real_e_no_sintetico():
    # a MESMA função calcula phi no real e no sintético
    df = pd.read_parquet(REAL, columns=["data", "era5_precip_mm"])
    df["data"] = pd.to_datetime(df["data"]); df = df[df["data"].dt.year <= 2022]
    phi_estim, _ = sv.estimar_ar1_real(REAL)
    phi_func = sv.phi_lag1_mensal(df["data"], df["era5_precip_mm"])
    assert abs(phi_estim - round(phi_func, 4)) < 1e-6, "estimar_ar1_real não usa phi_lag1_mensal"
    # no sintético a mesma função devolve um phi_ef válido e sensível ao parâmetro
    base_lo, _ = sv.gerar_cenario(0.0, 0, seed=0, clim=CLIM_FAKE)
    base_hi, _ = sv.gerar_cenario(0.95, 0, seed=0, clim=CLIM_FAKE)
    phi_lo = sv.phi_lag1_mensal(base_lo["data"], base_lo["era5_precip_mm"])
    phi_hi = sv.phi_lag1_mensal(base_hi["data"], base_hi["era5_precip_mm"])
    assert -1 <= phi_lo <= 1 and -1 <= phi_hi <= 1
    assert phi_hi > phi_lo + 0.3, "phi_ef não cresce com o phi do gerador"


@pytest.mark.skipif(not REAL.exists(), reason="dataset real ainda não gerado")
def test_gemeo_reproduz_media_e_desvio_mensal_real_dentro_de_10pct():
    clim = sv.calibrar_clima_real(REAL)
    phi_real, _ = sv.estimar_ar1_real(REAL)
    tau = sv.calibrar_tau_real(REAL)
    rm = (pd.read_parquet(REAL, columns=["data", "era5_precip_mm"])
          .assign(data=lambda d: pd.to_datetime(d["data"])))
    rm = rm[rm.data.dt.year <= 2022].set_index("data")["era5_precip_mm"].resample("ME").sum()
    real_mean, real_std = float(rm.mean()), float(rm.std())
    sigma_anom = sv.calibrar_sigma_anom(clim, phi_real, tau["tau"], real_std)

    means, stds = [], []
    for seed in range(5):
        base, _ = sv.gerar_cenario(phi_real, 0, seed, clim, sigma_anom=sigma_anom,
                                   tau=tau["tau"], ruido_afluencia=tau["sigma_mult"],
                                   normalizar_media=True)
        tm = base.set_index("data")["era5_precip_mm"].resample("ME").sum()
        means.append(tm.mean()); stds.append(tm.std())
    import numpy as np
    assert abs(np.mean(means) - real_mean) / real_mean < 0.10, "média mensal do gêmeo fora de 10%"
    assert abs(np.mean(stds) - real_std) / real_std < 0.10, "desvio mensal do gêmeo fora de 10%"
