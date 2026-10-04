# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import utils_v2 as u

RAIZ = Path(__file__).resolve().parents[2]
CSV_TESTE = RAIZ / "reports_v2" / "final" / "antecedencia_teste.csv"
CSV_VALIDACAO = RAIZ / "reports_v2" / "resultados" / "antecedencia_pareada_por_episodio.csv"

st.title("Retrospectiva das crises conhecidas")
st.caption("As crises de 2003-04, 2013-16 e 2021-22 usam previsões fora da amostra (out-of-fold) da "
           "validação. A crise de 2025-26 usa o teste com o modelo congelado (o mesmo cálculo da "
           "figura 16). Em todos os casos é o XGBRes residual contra o real e o B3.")

h = st.selectbox("Horizonte", [90, 60, 30], index=0)


@st.cache_data(show_spinner="Calculando a retrospectiva de validação...")
def _oof(h):
    return u.backtest_oof(u.carregar_dataset(), h)


@st.cache_data(show_spinner="Calculando a retrospectiva do teste (modelo congelado)...")
def _teste(h):
    return u.teste_congelado(u.carregar_dataset(), h)


CRISES = {"2003-04": ("2003-01-01", "2005-06-30"), "2013-16": ("2013-06-01", "2016-06-30"),
          "2021-22": ("2021-01-01", "2022-06-30"), "2025-26": ("2025-01-01", "2026-10-02")}
crise = st.radio("Episódio", list(CRISES.keys()), horizontal=True)
ini, fim = [pd.Timestamp(x) for x in CRISES[crise]]
eh_teste = crise == "2025-26"
origem = "teste, modelo congelado" if eh_teste else "validação"

oof = _teste(h) if eh_teste else _oof(h)
oof = oof.copy()
oof["data_alvo"] = pd.to_datetime(oof["data"]) + pd.to_timedelta(h, "D")
d = oof[(oof["data_alvo"] >= ini) & (oof["data_alvo"] <= fim)]

if len(d) == 0:
    st.warning("Sem dados para esse episódio neste horizonte.")
else:
    fig = go.Figure()
    for nome, lo, hi, cor in u.FAIXAS:
        fig.add_hrect(y0=lo, y1=hi, fillcolor=cor, opacity=0.08, line_width=0)
    fig.add_trace(go.Scatter(x=d["data_alvo"], y=d["real"], name="real", line=dict(color="#1f3b73", width=2)))
    if eh_teste and "vol_t" in d.columns:
        fig.add_trace(go.Scatter(x=d["data_alvo"], y=d["vol_t"], name="B1 (persistência)",
                                 line=dict(color="#7f7f7f", dash="dot")))
    fig.add_trace(go.Scatter(x=d["data_alvo"], y=d["B3"], name="B3", line=dict(color="#ff7f0e", dash="dash")))
    fig.add_trace(go.Scatter(x=d["data_alvo"], y=d["XGBRes"], name="XGBRes", line=dict(color="#d1495b")))
    fig.add_hline(y=40, line_dash="dot", line_color="#d62728")
    fig.update_layout(height=460, yaxis_title="Volume útil (%)", xaxis_title=f"Data alvo (t+{h})",
                      title=f"Crise {crise}: previsto (t+{h}) vs real ({origem})",
                      legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Origem dos dados: {origem}.")

st.subheader("Primeiro alerta (limiar 40%)")
if eh_teste:
    at = pd.read_csv(CSV_TESTE)
    sel = at[(at["limiar"] == 40) & (at["horizonte"] == h)].copy()
    sel["episodio_inicio"] = pd.to_datetime(sel["episodio_inicio"])
    sel = sel[(sel["episodio_inicio"] >= ini) & (sel["episodio_inicio"] <= fim)]
    if sel.empty:
        st.write("sem registro")
    else:
        disp = sel.sort_values(["episodio_inicio", "modelo"])[
            ["episodio_inicio", "modelo", "primeiro_alerta", "antecedencia_dias"]].copy()
        disp["episodio_inicio"] = disp["episodio_inicio"].dt.strftime("%Y-%m-%d")
        disp = disp.rename(columns={"episodio_inicio": "Episódio (entrada)", "modelo": "Modelo",
                                    "primeiro_alerta": "Primeiro alerta",
                                    "antecedencia_dias": "Antecedência (dias)"})
        st.dataframe(disp, hide_index=True, use_container_width=True)
    st.caption("Data do primeiro alerta e antecedência do teste (modelo congelado), lidas de "
               "reports_v2/final/antecedencia_teste.csv.")
else:
    av = pd.read_csv(CSV_VALIDACAO)
    sel = av[(av["limiar"] == 40) & (av["horizonte"] == h)].copy()
    sel["episodio_inicio"] = pd.to_datetime(sel["episodio_inicio"])
    sel = sel[(sel["episodio_inicio"] >= ini) & (sel["episodio_inicio"] <= fim)]
    if sel.empty:
        st.write("sem registro")
    else:
        longo = sel.sort_values("episodio_inicio").melt(
            id_vars=["episodio_inicio"], value_vars=["ant_XGBRes", "ant_B3"],
            var_name="Modelo", value_name="Antecedência (dias)")
        longo["Modelo"] = longo["Modelo"].map({"ant_XGBRes": "XGBRes", "ant_B3": "B3"})
        longo["episodio_inicio"] = longo["episodio_inicio"].dt.strftime("%Y-%m-%d")
        longo = longo.rename(columns={"episodio_inicio": "Episódio (entrada)"})
        st.dataframe(longo[["Episódio (entrada)", "Modelo", "Antecedência (dias)"]],
                     hide_index=True, use_container_width=True)
    st.caption("Antecedência em dias da validação (antecedência pareada por episódio, "
               "reports_v2/resultados/antecedencia_pareada_por_episodio.csv). Esse arquivo não traz a "
               "data do primeiro alerta, por isso a validação aparece só em dias.")
