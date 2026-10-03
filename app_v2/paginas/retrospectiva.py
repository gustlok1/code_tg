# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import utils_v2 as u

st.title("Retrospectiva das crises conhecidas (validação)")
st.caption("Previsões fora da amostra (out-of-fold) do XGBRes residual contra o real e o B3, "
           "nas crises históricas. Tudo da validação (2003-2022); o teste 2023+ não entra aqui.")

h = st.selectbox("Horizonte", [90, 60, 30], index=0)


@st.cache_data(show_spinner="Calculando a retrospectiva...")
def _oof(h):
    return u.backtest_oof(u.carregar_dataset(), h)


oof = _oof(h)
oof["data_alvo"] = pd.to_datetime(oof["data"]) + pd.to_timedelta(h, "D")

CRISES = {"2003-04": ("2003-01-01", "2005-06-30"), "2013-16": ("2013-06-01", "2016-06-30"),
          "2021-22": ("2021-01-01", "2022-06-30"), "2025-26": ("2025-01-01", "2026-10-02")}
crise = st.radio("Episódio", list(CRISES.keys()), horizontal=True)
ini, fim = [pd.Timestamp(x) for x in CRISES[crise]]
d = oof[(oof["data_alvo"] >= ini) & (oof["data_alvo"] <= fim)]

if len(d) == 0:
    st.warning("Esse episódio está fora da validação (ex.: 2025-26 está no teste). Veja a página inicial.")
else:
    fig = go.Figure()
    for nome, lo, hi, cor in u.FAIXAS:
        fig.add_hrect(y0=lo, y1=hi, fillcolor=cor, opacity=0.08, line_width=0)
    fig.add_trace(go.Scatter(x=d["data_alvo"], y=d["real"], name="real", line=dict(color="#1f3b73", width=2)))
    fig.add_trace(go.Scatter(x=d["data_alvo"], y=d["B3"], name="B3", line=dict(color="#ff7f0e", dash="dash")))
    fig.add_trace(go.Scatter(x=d["data_alvo"], y=d["XGBRes"], name="XGBRes", line=dict(color="#d1495b")))
    fig.add_hline(y=40, line_dash="dot", line_color="#d62728")
    fig.update_layout(height=460, yaxis_title="Volume útil (%)", xaxis_title=f"Data alvo (t+{h})",
                      title=f"Crise {crise} — previsto (t+{h}) vs real", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)
