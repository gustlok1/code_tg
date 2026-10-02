# -*- coding: utf-8 -*-
"""
sintetico_v2.py — Gerador de cenários SINTÉTICOS (Exp 1) + modelagem controlada.

Objetivo: ambiente controlado para mostrar (a) que o pipeline recupera um sinal
conhecido, (b) que o ML só supera o B3 quando a chuva futura tem componente previsível
a partir do passado (controlado por φ) e (c) como o desempenho muda com o nº de crises N.

NADA aqui mistura valores reais no dataset sintético: a climatologia real é usada só
para CALIBRAR a forma da chuva (prob. de ocorrência e gama por mês) e para estimar φ/σ
reais; os valores sintéticos são gerados de novo, com seed por cenário.

Esquema de saída idêntico ao dataset real → dataset_v2.construir_features/alvos rodam
sem alteração. Reaproveita modelagem_v2 (folds, purga, B3) e transformadores (SPI).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import gamma as gamma_dist
from xgboost import XGBRegressor

import dataset_v2 as dv
import modelagem_v2 as mv
import transformadores as tr

from pathlib import Path

DIR_SINTETICO = Path(__file__).resolve().parents[1] / "data" / "sintetico"


def caminho_cenario(nome: str) -> Path:
    """Caminho de saída de um cenário — SEMPRE contém 'sintetico' (marcação)."""
    p = DIR_SINTETICO / nome
    assert "sintetico" in str(p).lower()
    return p


CAP_HM3 = 981.56                     # capacidade útil do Sistema Cantareira (hm³)
FLOOR_FRAC = -0.30                   # volume pode ir a −30% (reserva técnica)
AREA_KM2 = 2280.0                    # área de drenagem aproximada
RUNOFF_COEF = 0.22
XGB_PARAMS = dict(max_depth=3, n_estimators=300, learning_rate=0.03, reg_lambda=15.0,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                  random_state=42, n_jobs=-1, objective="reg:squarederror")


# ============================================================ calibração (real)
def calibrar_clima_real(dataset_real_path) -> dict:
    """Prob. de ocorrência e parâmetros gama da chuva por mês, da ERA5 real (1984-2022)."""
    df = pd.read_parquet(dataset_real_path, columns=["data", "era5_precip_mm"])
    df["data"] = pd.to_datetime(df["data"])
    df = df[df["data"].dt.year <= 2022]
    mes = df["data"].dt.month.values
    p = df["era5_precip_mm"].values
    pocc, gshape, gscale = {}, {}, {}
    for m in range(1, 13):
        pm = p[mes == m]
        molhado = pm[pm > 1.0]
        pocc[m] = float((pm > 1.0).mean())
        if len(molhado) > 10:
            sh, _loc, sc = gamma_dist.fit(molhado, floc=0)
        else:
            sh, sc = 1.0, float(np.mean(molhado) if len(molhado) else 5.0)
        gshape[m], gscale[m] = float(sh), float(sc)
    return {"pocc": pocc, "gshape": gshape, "gscale": gscale}


def phi_lag1_mensal(datas, precip) -> float:
    """Persistência EFETIVA: autocorrelação lag-1 das anomalias mensais (log) da chuva.
    MESMA função usada no real e no sintético (garante comparabilidade do φ_ef)."""
    s = pd.Series(np.asarray(precip, dtype=float), index=pd.to_datetime(datas)).resample("ME").sum()
    clim = s.groupby(s.index.month).transform("mean")
    a = np.log((s + 1.0) / (clim + 1.0)).values
    x0, x1 = a[:-1], a[1:]
    return float(np.corrcoef(x0, x1)[0, 1])


def sigma_resid_ar1(datas, precip) -> float:
    """Desvio do resíduo do AR(1) das anomalias mensais (log)."""
    s = pd.Series(np.asarray(precip, dtype=float), index=pd.to_datetime(datas)).resample("ME").sum()
    clim = s.groupby(s.index.month).transform("mean")
    a = np.log((s + 1.0) / (clim + 1.0)).values
    x0, x1 = a[:-1], a[1:]
    phi = float(np.corrcoef(x0, x1)[0, 1])
    return float(np.std(x1 - phi * x0, ddof=1))


def estimar_ar1_real(dataset_real_path) -> tuple[float, float]:
    """φ e σ reais da chuva ERA5 (1984-2022), pelas funções acima."""
    df = pd.read_parquet(dataset_real_path, columns=["data", "era5_precip_mm"])
    df["data"] = pd.to_datetime(df["data"])
    df = df[df["data"].dt.year <= 2022]
    return (round(phi_lag1_mensal(df["data"], df["era5_precip_mm"]), 4),
            round(sigma_resid_ar1(df["data"], df["era5_precip_mm"]), 4))


def calibrar_tau_real(dataset_real_path) -> dict:
    """Ajusta a constante de tempo τ do reservatório linear à afluência real do SAR
    (minimiza o erro entre a afluência simulada a partir da chuva ERA5 e a real,
    1984-2022). Retorna τ, ganho k, RMSE, erro relativo e σ do resíduo multiplicativo."""
    df = pd.read_parquet(dataset_real_path,
                         columns=["data", "era5_precip_mm", "afluencia_m3s_sistema"])
    df["data"] = pd.to_datetime(df["data"])
    df = df[(df["data"].dt.year <= 2022)].dropna().reset_index(drop=True)
    precip = df["era5_precip_mm"].values
    infl_real = df["afluencia_m3s_sistema"].values
    melhor = {"rmse": np.inf}
    for tau in range(8, 121, 2):
        alpha = np.exp(-1.0 / tau)
        resp = np.zeros(len(precip))
        for t in range(1, len(precip)):
            resp[t] = alpha * resp[t - 1] + (1 - alpha) * precip[t]
        denom = float(np.dot(resp, resp))
        if denom <= 0:
            continue
        k = float(np.dot(resp, infl_real) / denom)          # ganho por mínimos quadrados
        pred = k * resp
        rmse = float(np.sqrt(np.mean((infl_real - pred) ** 2)))
        if rmse < melhor["rmse"]:
            # σ do resíduo multiplicativo (log), só em linhas com afluência e previsão
            # positivas (a afluência do SAR tem alguns negativos espúrios)
            m = (infl_real > 0.5) & (pred > 0.5)
            sig = float(np.std(np.log(infl_real[m] / pred[m]), ddof=1))
            melhor = {"tau": int(tau), "k": round(k, 4), "rmse": round(rmse, 3),
                      "erro_rel": round(rmse / float(np.mean(infl_real)), 4),
                      "sigma_mult": round(sig, 4)}
    return melhor


def calibrar_sigma_anom(clim: dict, phi: float, tau: float, std_alvo: float,
                        anos: int = 40, inicio: str = "1984-01-01", seed: int = 0) -> float:
    """Encontra o σ da anomalia que casa o DESVIO mensal da chuva com `std_alvo` (real).
    O σ do AR(1) log não é diretamente o σ do multiplicador por causa do ruído diário
    (ocorrência Bernoulli × gama), então calibramos numericamente."""
    melhor = (0.0, np.inf)
    for s in np.linspace(0.0, 0.7, 15):
        base, _ = gerar_cenario(phi, 0, seed, clim, sigma_anom=float(s), tau=tau,
                                normalizar_media=True, anos=anos, inicio=inicio)
        std = base.set_index("data")["era5_precip_mm"].resample("ME").sum().std()
        erro = abs(std - std_alvo)
        if erro < melhor[1]:
            melhor = (float(s), erro)
    return round(melhor[0], 4)


# ============================================================ gerador de cenário
def gerar_cenario(phi: float, n_secas: int, seed: int, clim: dict,
                  anos: int = 40, inicio: str = "1984-01-01",
                  sigma_anom: float | None = None, tau: float | None = None,
                  ruido_afluencia: float = 0.0, normalizar_media: bool = False) -> tuple:
    """Gera 40 anos diários. Retorna (base_schema, componentes) — 'componentes' traz
    V, afluência, retirada efetiva, evaporação e vertimento p/ o teste de conservação.

    Parâmetros opcionais (defaults reproduzem o comportamento original):
      sigma_anom: desvio da inovação da anomalia mensal AR(1) (default 0.5).
      tau: constante de tempo do reservatório linear (default: sorteada em [30,60]).
      ruido_afluencia: desvio do ruído multiplicativo log-normal na afluência (default 0).
      normalizar_media: se True, normaliza o multiplicador exp(anomalia) p/ média 1
                        (preserva a média mensal da chuva — usado no gêmeo calibrado)."""
    rng = np.random.default_rng(seed)
    datas = pd.date_range(inicio, periods=anos * 365, freq="D")
    n = len(datas)
    mes = datas.month.values
    doy = datas.dayofyear.values
    nmes = anos * 12

    # --- anomalia mensal AR(1) + secas injetadas ---
    sigma = 0.5 if sigma_anom is None else float(sigma_anom)
    a = np.zeros(nmes)
    eps = rng.normal(0, sigma, nmes)
    for t in range(1, nmes):
        a[t] = phi * a[t - 1] + eps[t]
    secas = []
    for _ in range(n_secas):
        dur = int(rng.integers(6, 13))                     # 6 a 12 meses
        ini = int(rng.integers(12, max(13, nmes - dur)))
        fator = float(rng.uniform(0.4, 0.6))
        a[ini:ini + dur] += np.log(fator)                  # mudança de regime (seca)
        secas.append((ini, dur, fator))
    idx_mes = (datas.year - datas.year[0]) * 12 + (datas.month - 1)
    a_dia = a[idx_mes]

    # --- chuva diária: ocorrência Bernoulli × gama × exp(anomalia) ---
    pocc = np.array([clim["pocc"][m] for m in mes])
    occ = rng.random(n) < pocc
    sh = np.array([clim["gshape"][m] for m in mes])
    sc = np.array([clim["gscale"][m] for m in mes])
    amount = rng.gamma(sh, sc)
    mult = np.exp(a_dia)
    if normalizar_media:
        mult = mult / mult.mean()                 # preserva a média (gêmeo calibrado)
    precip = np.where(occ, amount, 0.0) * mult
    precip = np.clip(precip, 0, 400)

    # --- temperatura / ET0 / radiação sazonais (preenchem o schema) ---
    season = np.sin(2 * np.pi * (doy - 15) / 365.0)
    tmean = 21.0 + 4.0 * season + rng.normal(0, 1.2, n)
    tmax = tmean + 5 + rng.normal(0, 1, n)
    tmin = tmean - 5 + rng.normal(0, 1, n)
    et0 = np.clip(3.5 + 1.6 * season + rng.normal(0, 0.3, n), 0.2, None)
    rad = np.clip(15 + 8 * season + rng.normal(0, 1.5, n), 1, None)

    # --- afluência: reservatório linear sobre o escoamento (τ 30–60 d, ou fixo) ---
    tau = float(rng.uniform(30, 60)) if tau is None else float(tau)
    alpha = np.exp(-1.0 / tau)
    runoff_hm3 = precip * RUNOFF_COEF * AREA_KM2 * 1e-3     # mm → hm³
    infl = np.zeros(n)
    for t in range(1, n):
        infl[t] = alpha * infl[t - 1] + (1 - alpha) * runoff_hm3[t]
    if ruido_afluencia > 0:                                 # ruído multiplicativo log-normal
        infl = infl * np.exp(rng.normal(0.0, ruido_afluencia, n))

    # --- balanço hídrico com faixas da Resolução 925 ---
    cap, floor = CAP_HM3, FLOOR_FRAC * CAP_HM3
    retirada_base = 0.97 * float(np.mean(infl[365:]))       # demanda ~ afluência média
    evap = np.clip(0.25 * (1 + 0.6 * season), 0, None)      # evaporação sazonal (hm³/d)

    V = np.empty(n); V[0] = 0.70 * cap
    ret_eff = np.zeros(n); vert = np.zeros(n)
    for t in range(n - 1):
        pct = 100.0 * V[t] / cap
        f = 1.0 if pct > 60 else 0.85 if pct > 40 else 0.70 if pct > 30 else 0.55 if pct > 20 else 0.40
        demanda = retirada_base * f
        ret_eff[t] = min(demanda, max(0.0, V[t] + infl[t] - evap[t] - floor))
        v_raw = V[t] + infl[t] - ret_eff[t] - evap[t]
        vert[t] = max(0.0, v_raw - cap)
        V[t + 1] = v_raw - vert[t]
    # último passo (diagnóstico de saída)
    pct = 100.0 * V[-1] / cap
    f = 1.0 if pct > 60 else 0.85 if pct > 40 else 0.70 if pct > 30 else 0.55 if pct > 20 else 0.40
    ret_eff[-1] = min(retirada_base * f, max(0.0, V[-1] + infl[-1] - evap[-1] - floor))
    vert[-1] = max(0.0, V[-1] + infl[-1] - ret_eff[-1] - evap[-1] - cap)

    deflu_hm3 = ret_eff + vert
    hm3d_to_m3s = 1e6 / 86400.0

    base = pd.DataFrame({
        "data": datas,
        "vol_util_hm3_sistema": V,
        "capacidade_hm3_sistema": cap,
        "vol_util_pct_sistema": 100.0 * V / cap,
        "afluencia_m3s_sistema": infl * hm3d_to_m3s,
        "defluencia_m3s_sistema": deflu_hm3 * hm3d_to_m3s,
        "n_reservatorios": 4,
        "era5_precip_mm": precip, "era5_et0_mm": et0, "era5_tmean_c": tmean,
        "era5_tmax_c": tmax, "era5_tmin_c": tmin, "era5_rad_mj": rad,
        "power_precip_mm": precip, "power_tmean_c": tmean, "power_tmax_c": tmax,
        "power_tmin_c": tmin, "power_rad_mj": rad, "power_rh_pct": 70 + 10 * season,
        "power_wind_ms": 2.0, "power_pressure_kpa": 92.0,
    })
    comp = pd.DataFrame({"data": datas, "V": V, "infl": infl, "ret_eff": ret_eff,
                         "evap": evap, "vert": vert, "precip": precip})
    comp.attrs["secas"] = secas; comp.attrs["tau"] = tau; comp.attrs["cap"] = cap
    return base, comp


def construir_dataset(base: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Aplica dataset_v2 (sem alteração) p/ obter o dataset no MESMO schema do real."""
    base = base.sort_values("data").reset_index(drop=True)
    feats = dv.construir_features(base)
    alvos = dv.construir_alvos(base, cfg["horizontes"], cfg["limiar_crise"],
                               cfg["limiar_crise_sensibilidade"])
    return base.merge(feats, on="data").merge(alvos, on="data")


# ============================================================ modelagem controlada
FEATS = mv.CLIMA_ERA5 + mv.ESTADO


def _features(dataset, train_mask):
    spi = tr.aplicar_indicadores(dataset, pd.Series(train_mask, index=dataset.index),
                                 col_data="data", col_precip="era5_precip_mm",
                                 col_et0="era5_et0_mm", escalas=(3, 6, 12))
    X = dataset[FEATS].join(spi)
    return X, FEATS + list(spi.columns)


def modelar_sintetico(dataset, cfg, h, modo="normal", seed=0):
    """modo: 'normal' | 'shuffle' (alvo embaralhado) | 'oraculo' (chuva futura t..t+h,
    DIAGNÓSTICO, não é modelo de previsão). Retorna (skill_B3_medio, oof_df)."""
    folds = mv.folds_config(cfg)
    alvo = f"vol_pct_t{h}"
    skills, oof_parts = [], []
    precip_fut = None
    if modo == "oraculo":
        p = dataset["era5_precip_mm"].values
        precip_fut = np.array([np.sum(p[i + 1:i + 1 + h]) if i + 1 + h <= len(p) else np.nan
                               for i in range(len(p))])
    for fo in folds:
        trm, vam = mv.mascaras_fold(dataset["data"], fo["ini"], fo["fim"], h)
        X, cols = _features(dataset, trm)
        if modo == "oraculo":
            X = X.assign(oraculo_chuva_fut=precip_fut); cols = cols + ["oraculo_chuva_fut"]
        ok = X[cols].notna().all(axis=1).values & dataset[alvo].notna().values
        tr_ok, va_ok = trm & ok, vam & ok
        if tr_ok.sum() < 200 or va_ok.sum() < 30:
            continue
        delta = mv.ajustar_B3(dataset[tr_ok], h)
        b3_tr = mv.prever_B3(dataset[tr_ok], h, delta)
        b3_va = mv.prever_B3(dataset[va_ok], h, delta)
        r_tr = dataset.loc[tr_ok, alvo].values - b3_tr
        if modo == "shuffle":
            r_tr = np.random.default_rng(seed + hash(fo["nome"]) % 997).permutation(r_tr)
        mod = XGBRegressor(**XGB_PARAMS); mod.fit(X[tr_ok], r_tr)
        pred_vol = b3_va + mod.predict(X[va_ok])
        real = dataset.loc[va_ok, alvo].values
        mae = np.mean(np.abs(real - pred_vol)); mae_b3 = np.mean(np.abs(real - b3_va))
        skills.append(1 - mae / mae_b3 if mae_b3 > 0 else np.nan)
        oof_parts.append(pd.DataFrame({
            "data": dataset.loc[va_ok, "data"].values, "real": real,
            "vol_t": dataset.loc[va_ok, "vol_pct"].values, "B3": b3_va, "XGBRes": pred_vol}))
    oof = (pd.concat(oof_parts, ignore_index=True).sort_values("data").reset_index(drop=True)
           if oof_parts else pd.DataFrame())
    return (float(np.nanmean(skills)) if skills else np.nan), oof
