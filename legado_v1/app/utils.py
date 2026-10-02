from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import streamlit as st

ROOT = Path(__file__).parent.parent

MODEL_PATH = ROOT / "resultados_v2_threshold_otimizado" / "modelo_crise_hidrica_y90.pkl"

MODEL_PATHS = {
    "y30": ROOT / "resultados_v2_threshold_otimizado" / "modelo_crise_hidrica_y30.pkl",
    "y60": ROOT / "resultados_v2_threshold_otimizado" / "modelo_crise_hidrica_y60.pkl",
    "y90": ROOT / "resultados_v2_threshold_otimizado" / "modelo_crise_hidrica_y90.pkl",
}

HORIZON_LABELS = {
    "y30": "30 dias",
    "y60": "60 dias",
    "y90": "90 dias",
}

PARQUET_PATH = ROOT / "algoritmos" / "data" / "features" / "inmet_sp_daily_labels.parquet"

FEATURES = [
    "PRECIP_DIARIA", "TMEAN", "TMAX", "TMIN", "URMEAN", "RAD_SUM",
    "PRESSAO_MED", "VENTO_MED", "RAJADA_MAX", "P7", "P15", "P30", "P90",
    "DRY_STREAK_CUR", "DRY30_MAX", "DRY90_MAX", "TMEAN_MA7", "TMEAN_MA30",
    "URMEAN_MA7", "URMEAN_MA30", "PRESSAO_MED_MA7", "SPI30_APRX",
]

FEATURE_LABELS = {
    "PRECIP_DIARIA":   "Precipitacao diaria (mm)",
    "TMEAN":           "Temperatura media (graus C)",
    "TMAX":            "Temperatura maxima (graus C)",
    "TMIN":            "Temperatura minima (graus C)",
    "URMEAN":          "Umidade relativa media (%)",
    "RAD_SUM":         "Radiacao global acumulada (kJ/m2)",
    "PRESSAO_MED":     "Pressao atmosferica media (mB)",
    "VENTO_MED":       "Velocidade do vento (m/s)",
    "RAJADA_MAX":      "Rajada maxima de vento (m/s)",
    "P7":              "Precipitacao acumulada 7 dias (mm)",
    "P15":             "Precipitacao acumulada 15 dias (mm)",
    "P30":             "Precipitacao acumulada 30 dias (mm)",
    "P90":             "Precipitacao acumulada 90 dias (mm)",
    "DRY_STREAK_CUR":  "Sequencia seca atual (dias)",
    "DRY30_MAX":       "Maior sequencia seca em 30 dias (dias)",
    "DRY90_MAX":       "Maior sequencia seca em 90 dias (dias)",
    "TMEAN_MA7":       "Temperatura media, media movel 7 dias (graus C)",
    "TMEAN_MA30":      "Temperatura media, media movel 30 dias (graus C)",
    "URMEAN_MA7":      "Umidade relativa, media movel 7 dias (%)",
    "URMEAN_MA30":     "Umidade relativa, media movel 30 dias (%)",
    "PRESSAO_MED_MA7": "Pressao atmosferica, media movel 7 dias (mB)",
    "SPI30_APRX":      "SPI-30 aproximado (z-score mensal)",
}

CRISES = [
    {"label": "Cantareira 2003-2004", "start": "2003-11-01", "end": "2004-04-30"},
    {"label": "Crise hidrica 2014-2015", "start": "2014-02-01", "end": "2015-10-31"},
    {"label": "Alerta hidrico 2021", "start": "2021-05-01", "end": "2021-11-30"},
]


@st.cache_resource
def carregar_modelo(target: str = "y90"):
    path = MODEL_PATHS.get(target, MODEL_PATH)
    pkg = joblib.load(path)
    return pkg["modelo"], float(pkg["threshold"])


@st.cache_data
def carregar_historico():
    df = pd.read_parquet(PARQUET_PATH)
    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_convert(None)
    return df


def validar_csv(df: pd.DataFrame) -> tuple[bool, list[str]]:
    erros = []
    if "timestamp" not in df.columns:
        erros.append("Coluna **timestamp** ausente. Formato esperado: YYYY-MM-DD.")
    else:
        convertidos = pd.to_datetime(df["timestamp"], errors="coerce")
        n_invalidos = convertidos.isna().sum()
        if n_invalidos > 0:
            erros.append(f"**{n_invalidos}** valores invalidos na coluna timestamp.")
    faltando = [f for f in FEATURES if f not in df.columns]
    if faltando:
        erros.append(f"Colunas de features ausentes ({len(faltando)}): `{', '.join(faltando)}`")
    return len(erros) == 0, erros


def classificar_risco(prob: float, threshold: float) -> tuple[str, str]:
    """Retorna (nivel_textual, hex_cor) baseado na probabilidade e threshold."""
    if prob >= threshold:
        return "Alto", "#d62728"
    elif prob >= threshold * 0.5:
        return "Medio", "#ff7f0e"
    return "Baixo", "#2ca02c"


def prever(modelo, threshold: float, df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES].fillna(0)
    probs = modelo.predict_proba(X)[:, 1]
    resultado = df[["timestamp"]].copy() if "timestamp" in df.columns else pd.DataFrame(index=df.index)
    resultado["probabilidade"] = probs
    resultado["prob_pct"] = (probs * 100).round(4)
    riscos = [classificar_risco(p, threshold) for p in probs]
    resultado["risco"] = [r[0] for r in riscos]
    resultado["cor"] = [r[1] for r in riscos]
    return resultado


def gerar_template_csv() -> str:
    header = "timestamp," + ",".join(FEATURES)
    row = "2024-01-01," + ",".join(["0.0"] * len(FEATURES))
    return header + "\n" + row + "\n"
