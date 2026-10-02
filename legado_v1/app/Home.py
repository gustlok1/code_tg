import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta
from utils import carregar_modelo, carregar_historico, FEATURES, FEATURE_LABELS, HORIZON_LABELS

st.set_page_config(
    page_title="Sistema de Previsao de Crise Hidrica",
    page_icon=None,
    layout="wide",
)

st.title("Sistema de Previsao de Crise Hidrica")
st.markdown(
    "Plataforma de suporte a decisao para antecipacao de crises hidricas "
    "no Sistema Cantareira, fundamentada em dados climaticos do INMET (2003 a 2024) "
    "e em modelos de aprendizado de maquina supervisionado."
)
st.markdown(
    "Trabalho de Graduacao. Fatec Santana de Parnaiba. Tecnologia em Ciencia de Dados."
)

st.divider()

with st.spinner("Carregando dados historicos..."):
    df_hist = carregar_historico()

df_hist["timestamp"] = pd.to_datetime(df_hist["timestamp"])
data_min = df_hist["timestamp"].min().date()
data_max = df_hist["timestamp"].max().date()
default_ini = data_max - timedelta(days=89)

ctrl_col1, ctrl_col2 = st.columns([1, 2])

with ctrl_col1:
    horizonte = st.selectbox(
        "Horizonte de previsao",
        options=["y90", "y60", "y30"],
        format_func=lambda x: HORIZON_LABELS[x],
        index=0,
        help=(
            "Define o numero de dias de antecedencia com que o modelo "
            "foi treinado para identificar o risco de crise."
        ),
    )

with ctrl_col2:
    intervalo = st.date_input(
        "Periodo de analise",
        value=(default_ini, data_max),
        min_value=data_min,
        max_value=data_max,
        help="Selecione o intervalo de datas do dataset historico a ser analisado.",
    )

modelo, threshold = carregar_modelo(horizonte)
n_dias_horizonte = int(horizonte[1:])

if isinstance(intervalo, (list, tuple)) and len(intervalo) == 2:
    d_ini = pd.Timestamp(intervalo[0])
    d_fim = pd.Timestamp(intervalo[1])
else:
    d_ini = pd.Timestamp(default_ini)
    d_fim = pd.Timestamp(data_max)

mask = (df_hist["timestamp"] >= d_ini) & (df_hist["timestamp"] <= d_fim)
df_periodo = df_hist[mask].copy()

if df_periodo.empty:
    st.warning("Nenhum dado disponivel para o periodo selecionado.")
    st.stop()

X = df_periodo[FEATURES].fillna(0)
probs = modelo.predict_proba(X)[:, 1]
prob_media = float(np.mean(probs))

# Normaliza pelo threshold: valor 1.0 significa que a probabilidade media
# atingiu ou superou o limiar de classificacao do modelo.
indice = min(prob_media / threshold, 1.0) if threshold > 0 else 0.0


def classificar_indice(v: float) -> tuple[str, str]:
    if v >= 0.70:
        return "Risco Alto", "#d62728"
    elif v >= 0.30:
        return "Risco Medio", "#ff7f0e"
    return "Risco Baixo", "#2ca02c"


nivel_texto, cor_hex = classificar_indice(indice)


def _gauge_indice(v: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(v, 4),
        number={
            "valueformat": ".4f",
            "font": {"size": 52, "color": cor_hex},
        },
        gauge={
            "axis": {
                "range": [0, 1],
                "tickvals": [0, 0.30, 0.70, 1.0],
                "ticktext": ["0", "0,30", "0,70", "1"],
                "tickfont": {"size": 12},
            },
            "bar": {"color": cor_hex, "thickness": 0.22},
            "steps": [
                {"range": [0.00, 0.30], "color": "#d6f5d6"},
                {"range": [0.30, 0.70], "color": "#fff4d6"},
                {"range": [0.70, 1.00], "color": "#ffd6d6"},
            ],
            "threshold": {
                "line": {"color": "#555555", "width": 2},
                "thickness": 0.75,
                "value": v,
            },
        },
        title={
            "text": (
                "Indice de Risco Hidrico<br>"
                "<span style='font-size:13px; color:#555'>"
                "Probabilidade media normalizada pelo limiar (0 a 1)"
                "</span>"
            )
        },
    ))
    fig.update_layout(height=320, margin=dict(t=70, b=10, l=40, r=40))
    return fig


st.divider()

left, right = st.columns([1.5, 1])

with left:
    st.plotly_chart(_gauge_indice(indice), use_container_width=True)

with right:
    st.markdown("#### Classificacao")
    st.markdown(
        f"<div style='"
        f"font-size:26px; font-weight:bold; color:{cor_hex}; "
        f"border-left: 5px solid {cor_hex}; padding-left: 12px; margin-bottom: 16px'>"
        f"{nivel_texto}"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("#### Resumo do periodo")
    c1, c2, c3 = st.columns(3)
    c1.metric("Dias analisados", f"{len(df_periodo):,}")
    c2.metric("Indice de risco", f"{indice:.4f}")
    c3.metric("Horizonte", HORIZON_LABELS[horizonte])

    st.markdown("---")

    escala = {"Risco Baixo": "0 a 0,30", "Risco Medio": "0,30 a 0,70", "Risco Alto": "0,70 a 1"}
    st.markdown("#### Escala de referencia")
    for nivel, faixa in escala.items():
        cor_ref = classificar_indice({"Risco Baixo": 0.0, "Risco Medio": 0.5, "Risco Alto": 1.0}[nivel])[1]
        st.markdown(
            f"<span style='color:{cor_ref}; font-weight:bold'>{nivel}</span>: indice {faixa}",
            unsafe_allow_html=True,
        )

st.divider()

ini_fmt = d_ini.strftime("%d/%m/%Y")
fim_fmt = d_fim.strftime("%d/%m/%Y")
prob_pct = prob_media * 100

st.markdown(
    f"Com base nos dados do periodo de **{ini_fmt}** a **{fim_fmt}**, "
    f"o modelo identifica probabilidade media de **{prob_pct:.2f}%** "
    f"de ocorrencia de crise hidrica nos proximos **{n_dias_horizonte} dias**. "
    f"O nivel de risco classificado para o periodo e: **{nivel_texto}**."
)

with st.expander("Distribuicao de probabilidade diaria no periodo"):
    fig_serie = go.Figure()
    fig_serie.add_trace(go.Scatter(
        x=df_periodo["timestamp"],
        y=probs * 100,
        mode="lines",
        line=dict(color="#1f77b4", width=1.2),
        name="Probabilidade diaria (%)",
    ))
    fig_serie.add_hline(
        y=threshold * 100,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Limiar do modelo: {threshold*100:.2f}%",
        annotation_position="top right",
    )
    fig_serie.add_hline(
        y=prob_media * 100,
        line_dash="dot",
        line_color=cor_hex,
        annotation_text=f"Media do periodo: {prob_media*100:.4f}%",
        annotation_position="bottom right",
    )
    fig_serie.update_layout(
        xaxis_title="Data",
        yaxis_title="Probabilidade prevista (%)",
        height=280,
        margin=dict(t=20, b=30),
    )
    st.plotly_chart(fig_serie, use_container_width=True)

st.divider()
st.subheader("Variaveis de maior contribuicao")
st.markdown(
    "As tres variaveis climaticas com maior peso nas decisoes do modelo, "
    "conforme a importancia global calculada pelo algoritmo treinado."
)

if hasattr(modelo, "feature_importances_"):
    imp = pd.Series(modelo.feature_importances_, index=FEATURES).nlargest(3).sort_values()
    labels_imp = [FEATURE_LABELS.get(f, f) for f in imp.index]

    fig_imp = go.Figure(go.Bar(
        x=imp.values,
        y=labels_imp,
        orientation="h",
        marker_color=["#1f77b4", "#4a9fd4", "#7bbfe8"],
        text=[f"{v:.4f}" for v in imp.values],
        textposition="outside",
    ))
    fig_imp.update_layout(
        xaxis_title="Importancia relativa (criterio Gini)",
        height=220,
        margin=dict(t=10, b=30, l=10, r=80),
        xaxis=dict(range=[0, imp.max() * 1.25]),
    )
    st.plotly_chart(fig_imp, use_container_width=True)
else:
    st.info("O modelo selecionado nao expoe importancia de variaveis.")

with st.expander("Informacoes do modelo"):
    st.markdown(
        f"**Algoritmo:** {type(modelo).__name__}  \n"
        f"**Horizonte de treinamento:** {HORIZON_LABELS[horizonte]}  \n"
        f"**Threshold otimizado (classificacao binaria):** {threshold:.4f}  \n"
        f"**Numero de variaveis preditoras:** {len(FEATURES)}  \n"
        f"**Fonte dos dados:** INMET, Estacao Sao Paulo Mirante (A701), 2003 a 2024."
    )
