import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils import carregar_historico, FEATURES, FEATURE_LABELS, CRISES, HORIZON_LABELS

st.set_page_config(page_title="Visualizacao", page_icon=None, layout="wide")
st.title("Series Temporais Historicas")
st.markdown(
    "Explore as features climaticas do INMET (2003 a 2024) com os **periodos de crise hidrica** "
    "destacados em vermelho."
)

with st.spinner("Carregando dados historicos..."):
    df = carregar_historico()

df["timestamp"] = pd.to_datetime(df["timestamp"])
data_min = df["timestamp"].min().date()
data_max = df["timestamp"].max().date()

st.sidebar.header("Filtros")

intervalo = st.sidebar.date_input(
    "Periodo",
    value=(data_min, data_max),
    min_value=data_min,
    max_value=data_max,
)

features_disponiveis = [f for f in FEATURES if f in df.columns]
labels_disponiveis = [FEATURE_LABELS.get(f, f) for f in features_disponiveis]

features_sel_labels = st.sidebar.multiselect(
    "Features",
    options=labels_disponiveis,
    default=labels_disponiveis[:3],
)
label_para_col = {FEATURE_LABELS.get(f, f): f for f in features_disponiveis}
features_sel = [label_para_col[l] for l in features_sel_labels]

mostrar_crises = st.sidebar.checkbox("Destacar periodos de crise", value=True)

horizonte_label = st.sidebar.selectbox(
    "Sobrepor rotulos de pre-crise",
    options=["Nenhum", "y30 (30 dias)", "y60 (60 dias)", "y90 (90 dias)"],
    index=0,
    help="Marca os dias rotulados como pre-crise no dataset historico.",
)
horizonte_col = None
if horizonte_label != "Nenhum":
    horizonte_col = horizonte_label.split(" ")[0]

if isinstance(intervalo, (list, tuple)) and len(intervalo) == 2:
    d_ini = pd.Timestamp(intervalo[0])
    d_fim = pd.Timestamp(intervalo[1])
else:
    d_ini = pd.Timestamp(data_min)
    d_fim = pd.Timestamp(data_max)

mask = (df["timestamp"] >= d_ini) & (df["timestamp"] <= d_fim)
df_fil = df[mask].copy()

if df_fil.empty:
    st.warning("Nenhum dado no intervalo selecionado.")
    st.stop()

st.caption(
    f"Exibindo **{len(df_fil):,}** dias de {d_ini.date()} a {d_fim.date()}"
)

if not features_sel:
    st.info("Selecione ao menos uma feature no painel lateral.")
    st.stop()

PALETA = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

for i, feat in enumerate(features_sel):
    label = FEATURE_LABELS.get(feat, feat)
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_fil["timestamp"],
        y=df_fil[feat],
        mode="lines",
        name=label,
        line=dict(color=PALETA[i % len(PALETA)], width=1.2),
    ))

    if mostrar_crises:
        for crise in CRISES:
            c_ini = pd.Timestamp(crise["start"])
            c_fim = pd.Timestamp(crise["end"])
            if c_fim < d_ini or c_ini > d_fim:
                continue
            fig.add_vrect(
                x0=max(c_ini, d_ini),
                x1=min(c_fim, d_fim),
                fillcolor="red",
                opacity=0.10,
                line_width=0,
                annotation_text=crise["label"],
                annotation_position="top left",
                annotation=dict(font_size=10, font_color="red"),
            )

    if horizonte_col and horizonte_col in df_fil.columns:
        pos = df_fil[df_fil[horizonte_col] == 1]
        if not pos.empty:
            fig.add_trace(go.Scatter(
                x=pos["timestamp"],
                y=pos[feat],
                mode="markers",
                name=f"Pre-crise ({horizonte_col})",
                marker=dict(color="orange", size=6, symbol="circle-open", line=dict(width=1.5)),
                showlegend=(i == 0),
            ))

    fig.update_layout(
        title=label,
        xaxis_title="Data",
        yaxis_title=label,
        height=280,
        margin=dict(t=40, b=30, l=50, r=20),
        legend=dict(orientation="h", y=1.15),
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()
with st.expander("Estatisticas descritivas do periodo selecionado"):
    cols_stat = [f for f in features_sel if f in df_fil.columns]
    stats = df_fil[cols_stat].describe().T
    stats.index = [FEATURE_LABELS.get(c, c) for c in stats.index]
    st.dataframe(stats.round(3), use_container_width=True)
