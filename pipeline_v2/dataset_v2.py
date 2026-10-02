# -*- coding: utf-8 -*-
"""
dataset_v2.py — Núcleo (importável) da construção do dataset diário v2.

Separado de 01_build_dataset_v2.py porque um módulo cujo nome começa com dígito
não é importável por `import`; os testes (pytest) importam as funções daqui.

Contém: carga das fontes, média da bacia, calendário contínuo, FEATURES CAUSAIS,
ALVOS e detecção automática de episódios de crise. NENHUMA estatística de série
inteira (climatologia, z-score, SPI/SPEI) entra aqui — isso vive em transformadores.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ---- janelas de acumulação (dias) ----
JANELAS_ACC = [7, 30, 90, 180, 365]

# variáveis de clima por fonte (nome neutro no arquivo de pontos)
ERA5_VARS = ["precip_mm", "et0_mm", "tmean_c", "tmax_c", "tmin_c", "rad_mj"]
POWER_VARS = ["precip_mm", "tmean_c", "tmax_c", "tmin_c", "rad_mj",
              "rh_pct", "wind_ms", "pressure_kpa"]


# ----------------------------------------------------------------------------
# carga e média da bacia
# ----------------------------------------------------------------------------
def carregar_sistema(caminho) -> pd.DataFrame:
    df = pd.read_csv(caminho, parse_dates=["data"])
    return df.sort_values("data").reset_index(drop=True)


def media_bacia(caminho_pontos, prefixo: str, variaveis) -> pd.DataFrame:
    """Lê o CSV longo (1 linha por ponto por dia) e devolve a MÉDIA diária entre os
    pontos, com uma coluna por variável prefixada (ex.: era5_precip_mm)."""
    longo = pd.read_csv(caminho_pontos, parse_dates=["data"])
    cols = [c for c in variaveis if c in longo.columns]
    diario = (longo.groupby("data")[cols].mean().reset_index()
                   .rename(columns={c: f"{prefixo}_{c}" for c in cols}))
    return diario.sort_values("data").reset_index(drop=True)


def construir_calendario(sistema: pd.DataFrame, era5: pd.DataFrame, power: pd.DataFrame,
                         ano_inicial: int):
    """Calendário diário contínuo de {ano_inicial}-01-01 até a última data observada
    em qualquer fonte. Não inventa valores (deixa NaN). Retorna (base, buracos)."""
    ini = pd.Timestamp(f"{ano_inicial}-01-01")
    fim = max(sistema["data"].max(), era5["data"].max(), power["data"].max())
    calendario = pd.DataFrame({"data": pd.date_range(ini, fim, freq="D")})

    base = (calendario
            .merge(sistema, on="data", how="left")
            .merge(era5, on="data", how="left")
            .merge(power, on="data", how="left")
            .sort_values("data").reset_index(drop=True))

    # relatório de buracos por fonte (datas dentro do calendário sem dado)
    buracos = []
    checagens = {
        "sar_sistema": "vol_util_pct_sistema",
        "era5": "era5_precip_mm",
        "power": "power_precip_mm",
    }
    for fonte, col in checagens.items():
        if col not in base.columns:
            continue
        falt = base.loc[base[col].isna(), "data"]
        buracos.append({
            "fonte": fonte,
            "n_dias_ausentes": int(falt.shape[0]),
            "primeiro_ausente": falt.min().strftime("%Y-%m-%d") if len(falt) else "",
            "ultimo_ausente": falt.max().strftime("%Y-%m-%d") if len(falt) else "",
        })
    return base, pd.DataFrame(buracos)


# ----------------------------------------------------------------------------
# FEATURES CAUSAIS (dependem apenas de dados até o dia t)
# ----------------------------------------------------------------------------
def _acc(s: pd.Series, w: int) -> pd.Series:      # soma móvel (janela completa)
    return s.rolling(w, min_periods=w).sum()


def _ma(s: pd.Series, w: int) -> pd.Series:       # média móvel (janela completa)
    return s.rolling(w, min_periods=w).mean()


def construir_features(base: pd.DataFrame) -> pd.DataFrame:
    """Gera as features CAUSAIS. `base` deve vir ordenada por data, calendário contínuo.

    Toda feature usa apenas t e o passado (somas/médias móveis trailing e defasagens
    positivas). É isto que o teste de causalidade verifica.
    """
    base = base.sort_values("data").reset_index(drop=True)
    f = pd.DataFrame({"data": base["data"].values})

    # --- clima: chuva, ET0, balanço, temperatura (por fonte quando disponível) ---
    fontes_precip = [("era5", "era5_precip_mm"), ("power", "power_precip_mm")]
    et0_col = "era5_et0_mm"  # ET0 nativo só no ERA5; serve de demanda para o balanço
    tem_et0 = et0_col in base.columns

    if tem_et0:
        for w in JANELAS_ACC:
            f[f"et0_acc{w}"] = _acc(base[et0_col], w)

    for src, pcol in fontes_precip:
        if pcol not in base.columns:
            continue
        for w in JANELAS_ACC:
            f[f"{src}_precip_acc{w}"] = _acc(base[pcol], w)
            if tem_et0:
                # balanço hídrico = chuva − ET0 acumulados na mesma janela
                f[f"{src}_bal_acc{w}"] = f[f"{src}_precip_acc{w}"] - f[f"et0_acc{w}"]

    for src, tcol in [("era5", "era5_tmean_c"), ("power", "power_tmean_c")]:
        if tcol in base.columns:
            f[f"{src}_tmean_ma7"] = _ma(base[tcol], 7)
            f[f"{src}_tmean_ma30"] = _ma(base[tcol], 30)

    # --- estado do reservatório (do SAR) ---
    if "vol_util_pct_sistema" in base.columns:
        vol = base["vol_util_pct_sistema"]
        f["vol_pct"] = vol.values
        f["vol_var7"] = (vol - vol.shift(7)).values
        f["vol_var30"] = (vol - vol.shift(30)).values
    if "afluencia_m3s_sistema" in base.columns:
        afl = base["afluencia_m3s_sistema"]
        f["aflu_ma7"] = _ma(afl, 7).values
        f["aflu_ma30"] = _ma(afl, 30).values
        f["aflu_ma90"] = _ma(afl, 90).values
    if "defluencia_m3s_sistema" in base.columns:
        f["deflu_ma30"] = _ma(base["defluencia_m3s_sistema"], 30).values

    # --- sazonalidade (dia do ano em seno e cosseno) ---
    doy = base["data"].dt.dayofyear
    ano_dias = base["data"].dt.is_leap_year.map({True: 366, False: 365})
    f["doy_sin"] = np.sin(2 * np.pi * doy / ano_dias).values
    f["doy_cos"] = np.cos(2 * np.pi * doy / ano_dias).values

    # --- marco regulatório (Resolução ANA/DAEE 925, 29/05/2017) ---
    f["pos_2017"] = (base["data"] >= pd.Timestamp("2017-05-29")).astype(int).values

    return f


# ----------------------------------------------------------------------------
# ALVOS (olham o futuro; NaN no fim da série)
# ----------------------------------------------------------------------------
def construir_alvos(base: pd.DataFrame, horizontes, limiar: int,
                    limiar_sens: int) -> pd.DataFrame:
    base = base.sort_values("data").reset_index(drop=True)
    vol = base["vol_util_pct_sistema"]
    a = pd.DataFrame({"data": base["data"].values})
    for h in horizontes:
        futuro = vol.shift(-h)
        a[f"vol_pct_t{h}"] = futuro.values
        a[f"vol_var_t{h}"] = (futuro - vol).values
        crise = futuro.lt(limiar).astype("float"); crise[futuro.isna()] = np.nan
        a[f"crise_{h}"] = crise.values
        crise40 = futuro.lt(limiar_sens).astype("float"); crise40[futuro.isna()] = np.nan
        a[f"crise{limiar_sens}_{h}"] = crise40.values
    return a


# ----------------------------------------------------------------------------
# episódios automáticos de crise
# ----------------------------------------------------------------------------
def detectar_episodios(df_dia: pd.DataFrame, col_vol: str, limiar: int,
                       gap_max_dias: int, dur_min_dias: int) -> pd.DataFrame:
    """Trechos contínuos com vol < limiar, juntando lacunas < gap_max_dias e exigindo
    duração >= dur_min_dias. Calendário diário contínuo assumido."""
    d = df_dia.sort_values("data").reset_index(drop=True)
    abaixo = (d[col_vol] < limiar).fillna(False).values
    datas = d["data"].values

    # runs contíguos de 'abaixo'
    runs = []
    i, n = 0, len(abaixo)
    while i < n:
        if abaixo[i]:
            j = i
            while j + 1 < n and abaixo[j + 1]:
                j += 1
            runs.append([pd.Timestamp(datas[i]), pd.Timestamp(datas[j])])
            i = j + 1
        else:
            i += 1
    if not runs:
        return pd.DataFrame(columns=["inicio", "fim", "duracao_dias", "vol_min_pct"])

    # junta runs separados por lacuna < gap_max_dias
    merged = [runs[0]]
    for ini, fim in runs[1:]:
        if (ini - merged[-1][1]).days < gap_max_dias:
            merged[-1][1] = fim
        else:
            merged.append([ini, fim])

    linhas = []
    volser = d.set_index("data")[col_vol]
    for ini, fim in merged:
        dur = (fim - ini).days + 1
        if dur < dur_min_dias:
            continue
        janela = volser.loc[ini:fim]
        linhas.append({
            "inicio": ini.strftime("%Y-%m-%d"),
            "fim": fim.strftime("%Y-%m-%d"),
            "duracao_dias": int(dur),
            "vol_min_pct": round(float(janela.min()), 2) if janela.notna().any() else np.nan,
        })
    return pd.DataFrame(linhas)
