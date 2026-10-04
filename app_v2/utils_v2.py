# -*- coding: utf-8 -*-
"""
utils_v2.py: suporte do app Streamlit v2. Usa o MODELO CONGELADO (congelamento.json:
XGBRes residual sobre o B3) para prever o volume util do Sistema Cantareira em t+30/60/90,
com faixa de incerteza a partir dos residuos. Sem mencao a estacao A701.

Para uso OPERACIONAL (prever o futuro), o modelo congelado e retreinado em todo o historico
disponivel; o congelamento fixa conjunto, hiperparametros e thresholds, nao o periodo de treino.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
PIPE = ROOT / "pipeline_v2"
sys.path.insert(0, str(PIPE))

import analise_exp3 as ax           # noqa: E402
import modelagem_v2 as mv           # noqa: E402
import transformadores as tr        # noqa: E402
from xgboost import XGBRegressor    # noqa: E402

CONFIG = PIPE / "config.yaml"
CONGELAMENTO = ROOT / "reports_v2" / "final" / "congelamento.json"

FAIXAS = [("Normal", 60, 100, "#2ca02c"), ("Atenção", 40, 60, "#bcbd22"),
          ("Alerta", 30, 40, "#ff7f0e"), ("Restrição", 20, 30, "#d62728"),
          ("Reserva técnica", -30, 20, "#7f0000")]


def cfg():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def carregar_dataset():
    c = cfg()
    df = pd.read_parquet(ROOT / c["caminhos"]["processed_v2_dataset"])
    df["data"] = pd.to_datetime(df["data"])
    return df.sort_values("data").reset_index(drop=True)


def congelamento():
    if CONGELAMENTO.exists():
        return json.loads(CONGELAMENTO.read_text(encoding="utf-8"))
    return {"modelos": {"XGBRes": {str(h): {"config": {}} for h in (30, 60, 90)}},
            "thresholds_volume_alerta": {}}


def faixa_ana(v):
    for nome, lo, hi, cor in FAIXAS:
        if lo <= v < hi or (nome == "Normal" and v >= 100):
            return nome, cor
    return "Reserva técnica", "#7f0000"


def _features(df, mask_fit):
    spi = tr.aplicar_indicadores(df, pd.Series(mask_fit, index=df.index), col_data="data",
                                 col_precip="era5_precip_mm", col_et0="era5_et0_mm", escalas=(3, 6, 12))
    anom = tr.anomalias_sazonais(df, mask_fit, [f"era5_precip_acc{w}" for w in ax.ANOM_W])
    clima = ax._cols_clima("era5")
    cols = clima + list(ax.ESTADO) + list(spi.columns) + list(anom.columns)
    X = df[clima + list(ax.ESTADO)].join(spi).join(anom)
    return X, cols


def treinar_e_prever(df, h, config):
    """Treina o XGBRes congelado em todo o historico com alvo valido e preve as ultimas datas
    (onde t+h ainda nao existe). Retorna (previsoes_df, faixa_incerteza_pp)."""
    alvo = f"vol_pct_t{h}"
    mask_fit = df[alvo].notna().values & df["vol_pct"].notna().values
    X, cols = _features(df, mask_fit)
    ok = X[cols].notna().all(axis=1).values
    tr_ok = ok & df[alvo].notna().values
    delta = mv.ajustar_B3(df[tr_ok], h)
    b3_tr = mv.prever_B3(df[tr_ok], h, delta)
    r_tr = df.loc[tr_ok, alvo].values - b3_tr
    mod = XGBRegressor(**(config or {}), random_state=42, n_jobs=-1, objective="reg:squarederror")
    mod.fit(X[tr_ok], r_tr)
    # incerteza: percentis dos residuos de treino
    resid = df.loc[tr_ok, alvo].values - (b3_tr + mod.predict(X[tr_ok]))
    banda = float(np.percentile(np.abs(resid), 80))
    # prever as ultimas datas com features validas mas alvo ainda inexistente (futuro)
    prev_mask = ok & df[alvo].isna().values & df["vol_pct"].notna().values
    dprev = df[prev_mask].copy()
    if len(dprev) == 0:
        return pd.DataFrame(), banda
    b3p = mv.prever_B3(dprev, h, delta)
    pred = b3p + mod.predict(X[prev_mask])
    out = pd.DataFrame({"data_origem": dprev["data"].values,
                        "data_alvo": dprev["data"].values + pd.to_timedelta(h, "D"),
                        "vol_previsto": pred, "banda": banda})
    return out, banda


def previsao_atual(df):
    """Para a ultima observacao, devolve vol atual e previsao em 30/60/90 com faixa."""
    cong = congelamento()
    atual_data = df.loc[df["vol_pct"].notna(), "data"].max()
    vol_atual = float(df.loc[df["data"] == atual_data, "vol_pct"].iloc[0])
    linhas = []
    for h in (30, 60, 90):
        config = cong["modelos"]["XGBRes"].get(str(h), {}).get("config", {})
        prev, banda = treinar_e_prever(df, h, config)
        if len(prev):
            ult = prev.iloc[-1]
            v = float(ult["vol_previsto"])
            linhas.append({"horizonte": h, "data_alvo": pd.Timestamp(ult["data_alvo"]),
                           "vol_previsto": v, "lo": v - banda, "hi": v + banda,
                           "faixa": faixa_ana(v)[0]})
    return atual_data, vol_atual, pd.DataFrame(linhas)


def backtest_oof(df, h=90):
    """Previsoes OOF de validacao (XGBRes residual) para os anos das crises conhecidas."""
    c = cfg()
    _, oof = ax.validar_residual(df, c, h, src="era5")
    oof = oof.copy(); oof["data"] = pd.to_datetime(oof["data"])
    return oof[["data", "real", "B3", "XGBRes"]].sort_values("data")


def teste_congelado(df, h):
    """Previsoes do XGBRes CONGELADO no teste (>= teste_inicio), o MESMO calculo da figura 16
    (ax.testar_residual com a config gravada em congelamento.json). Nao reabre o teste nem refaz
    escolhas: a config ja esta congelada. Retorna data/real/vol_t/B3/XGBRes."""
    c = cfg()
    cong = congelamento()
    config = cong["modelos"]["XGBRes"].get(str(h), {}).get("config", {})
    oof = ax.testar_residual(df, c, h, config)
    oof = oof.copy(); oof["data"] = pd.to_datetime(oof["data"])
    return oof.sort_values("data")


def atualizar_por_api():
    """Roda a coleta (SAR+ERA5+POWER) e reconstroi o dataset. Retorna (ok, mensagem)."""
    try:
        subprocess.run([sys.executable, str(PIPE / "00_download_dados.py"), "--all"],
                       check=True, timeout=1800)
        subprocess.run([sys.executable, str(PIPE / "01_build_dataset_v2.py")],
                       check=True, timeout=600)
        return True, "Dados atualizados (SAR + ERA5 + NASA POWER) e dataset reconstruido."
    except Exception as e:  # noqa: BLE001
        return False, f"Falha na atualizacao: {e}"
