# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import utils_v2 as u

st.title("Previsão de crise hídrica no Sistema Cantareira")
st.caption("Volume útil do sistema (SAR/ANA) e previsão em 30, 60 e 90 dias. "
           "Modelo congelado (XGBRes residual sobre o B3). Fonte climática do modelo: ERA5-Land.")


@st.cache_data(show_spinner="Carregando dados e treinando o modelo congelado...")
def _carrega():
    df = u.carregar_dataset()
    atual_data, vol_atual, prev = u.previsao_atual(df)
    return df, atual_data, vol_atual, prev


df, atual_data, vol_atual, prev = _carrega()

c1, c2 = st.columns([1, 2])
with c1:
    nome, cor = u.faixa_ana(vol_atual)
    st.metric(f"Volume atual ({pd.Timestamp(atual_data).date()})",
              f"{vol_atual:.1f}".replace(".", ",") + "%")
    st.markdown(f"<div style='font-size:22px;font-weight:bold;color:{cor}'>Faixa: {nome}</div>",
                unsafe_allow_html=True)
    if st.button("Atualizar por API (SAR + ERA5 + POWER)"):
        with st.spinner("Baixando e reconstruindo..."):
            ok, msg = u.atualizar_por_api()
        (st.success if ok else st.error)(msg)
        st.cache_data.clear()
    st.caption("A atualização baixa SAR e ERA5-Land (usados pelo modelo) e também NASA POWER, "
               "este apenas para conferência da chuva. O modelo não usa POWER.")

with c2:
    st.subheader("Previsão por horizonte (com faixa de incerteza)")
    if len(prev):
        def _v(x):
            return f"{x:.1f}".replace(".", ",")
        tab = prev.copy()
        disp = pd.DataFrame({
            "Horizonte (dias)": tab["horizonte"].astype(int),
            "Data alvo": tab["data_alvo"].dt.strftime("%Y-%m-%d"),
            "Volume previsto (%)": tab["vol_previsto"].map(_v),
            "Mín (%)": tab["lo"].map(_v),
            "Máx (%)": tab["hi"].map(_v),
            "Faixa ANA/DAEE": tab["faixa"],
        })
        st.dataframe(disp, use_container_width=True, hide_index=True)

st.divider()
st.subheader("Volume do sistema e previsão")
hist = df[df["vol_pct"].notna()].tail(400)
fig = go.Figure()
for nome, lo, hi, cor in u.FAIXAS:
    fig.add_hrect(y0=lo, y1=hi, fillcolor=cor, opacity=0.08, line_width=0)
fig.add_trace(go.Scatter(x=hist["data"], y=hist["vol_pct"], mode="lines",
                         line=dict(color="#1f3b73", width=1.6), name="volume observado (%)"))
if len(prev):
    fig.add_trace(go.Scatter(x=prev["data_alvo"], y=prev["vol_previsto"], mode="markers+lines",
                             marker=dict(size=10, color="#d1495b"), name="previsão"))
    fig.add_trace(go.Scatter(
        x=list(prev["data_alvo"]) + list(prev["data_alvo"][::-1]),
        y=list(prev["hi"]) + list(prev["lo"][::-1]), fill="toself", fillcolor="rgba(209,73,91,0.2)",
        line=dict(width=0), name="faixa de incerteza", hoverinfo="skip"))
fig.update_layout(height=430, yaxis_title="Volume útil (%)", xaxis_title="Data",
                  legend=dict(orientation="h", y=1.1))
st.plotly_chart(fig, use_container_width=True)
st.caption("A faixa de incerteza é o percentil 80 do erro absoluto fora da amostra na validação "
           "(1984 a 2022). As faixas coloridas são as bandas operacionais ANA/DAEE.")
