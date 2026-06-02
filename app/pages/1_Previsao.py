import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils import (
    carregar_modelo, validar_csv, prever, gerar_template_csv,
    FEATURES, FEATURE_LABELS, HORIZON_LABELS,
)

st.set_page_config(page_title="Previsao", page_icon=None, layout="wide")
st.title("Previsao de Risco de Crise Hidrica")

horizonte = st.sidebar.selectbox(
    "Horizonte de previsao",
    options=["y90", "y60", "y30"],
    format_func=lambda x: HORIZON_LABELS[x],
    index=0,
    help="Com quantos dias de antecedencia o modelo preve o risco.",
)
modelo, threshold = carregar_modelo(horizonte)
st.sidebar.markdown(
    f"**Modelo:** {type(modelo).__name__}  \n"
    f"**Threshold:** `{threshold:.3f}`  \n"
    f"**Features:** {len(FEATURES)}"
)


def _cor_risco(prob: float) -> tuple[str, str]:
    if prob >= threshold:
        return "Alto", "#d62728"
    elif prob >= threshold * 0.5:
        return "Medio", "#ff7f0e"
    return "Baixo", "#2ca02c"


def _gauge(prob: float, threshold: float) -> go.Figure:
    nivel, cor = _cor_risco(prob)
    thr_pct = threshold * 100
    prob_pct = prob * 100
    escala_max = max(thr_pct * 3, 5.0)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob_pct, 4),
        number={"suffix": "%", "font": {"size": 36}},
        gauge={
            "axis": {"range": [0, escala_max], "ticksuffix": "%"},
            "bar": {"color": cor, "thickness": 0.25},
            "steps": [
                {"range": [0, thr_pct * 0.5], "color": "#d6f5d6"},
                {"range": [thr_pct * 0.5, thr_pct], "color": "#fff4d6"},
                {"range": [thr_pct, escala_max], "color": "#ffd6d6"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 3},
                "thickness": 0.8,
                "value": thr_pct,
            },
        },
        title={"text": f"Probabilidade de pre-crise<br><span style='font-size:14px'>Limiar = {thr_pct:.2f}%</span>"},
    ))
    fig.update_layout(height=280, margin=dict(t=60, b=10, l=30, r=30))
    return fig


def _importancias_top5(modelo, n: int = 5) -> go.Figure:
    if not hasattr(modelo, "feature_importances_"):
        return None
    imp = pd.Series(modelo.feature_importances_, index=FEATURES)
    top = imp.nlargest(n).sort_values()
    labels = [FEATURE_LABELS.get(f, f) for f in top.index]
    fig = go.Figure(go.Bar(
        x=top.values,
        y=labels,
        orientation="h",
        marker_color="#1f77b4",
    ))
    fig.update_layout(
        title=f"Top {n} features mais relevantes",
        xaxis_title="Importancia (Gini)",
        height=250,
        margin=dict(t=40, b=20, l=10, r=10),
    )
    return fig


def _exibir_resultado_unico(prob: float):
    nivel, cor = _cor_risco(prob)
    col_g, col_r = st.columns([1.2, 1])
    with col_g:
        st.plotly_chart(_gauge(prob, threshold), use_container_width=True)
    with col_r:
        st.markdown("### Classificacao de risco")
        st.markdown(
            f"<div style='font-size:28px; font-weight:bold; text-align:center; color:{cor}'>{nivel}</div>",
            unsafe_allow_html=True,
        )
        st.metric("Probabilidade", f"{prob*100:.4f}%")
        st.metric("Threshold do modelo", f"{threshold*100:.2f}%")
        atingido = "Acima do limiar" if prob >= threshold else "Abaixo do limiar"
        st.info(atingido)

    fig_imp = _importancias_top5(modelo)
    if fig_imp:
        st.plotly_chart(fig_imp, use_container_width=True)


aba_csv, aba_manual = st.tabs(["Upload CSV", "Insercao manual"])


with aba_csv:
    st.markdown(
        f"Faca upload de um CSV com `timestamp` + as **{len(FEATURES)} features** "
        f"climaticas para obter a previsao de risco com **{HORIZON_LABELS[horizonte]}** de antecedencia."
    )

    with st.expander("Baixar template CSV"):
        st.code(", ".join(["timestamp"] + FEATURES), language="text")
        st.download_button(
            "Baixar template.csv",
            data=gerar_template_csv(),
            file_name="template_previsao.csv",
            mime="text/csv",
        )

    arquivo = st.file_uploader("Selecione o arquivo CSV", type=["csv"], key="csv_upload")

    if arquivo is not None:
        try:
            df = pd.read_csv(arquivo)
        except Exception as e:
            st.error(f"Erro ao ler o arquivo: {e}")
            st.stop()

        ok, erros = validar_csv(df)
        if not ok:
            st.error("Validacao falhou:")
            for err in erros:
                st.markdown(f"- {err}")
        else:
            st.success(f"CSV valido: {len(df):,} linhas")
            with st.expander("Previa dos dados"):
                st.dataframe(df.head(10), use_container_width=True)

            df["timestamp"] = pd.to_datetime(df["timestamp"])
            resultado = prever(modelo, threshold, df)

            st.divider()
            n_alto  = (resultado["risco"] == "Alto").sum()
            n_medio = (resultado["risco"] == "Medio").sum()
            n_baixo = (resultado["risco"] == "Baixo").sum()
            c1, c2, c3 = st.columns(3)
            c1.metric("Alto", n_alto)
            c2.metric("Medio", n_medio)
            c3.metric("Baixo", n_baixo)

            exibir = resultado[["timestamp", "prob_pct", "risco"]].copy()
            exibir.columns = ["Data", "Probabilidade (%)", "Risco"]
            exibir["Data"] = exibir["Data"].dt.strftime("%Y-%m-%d")

            def _colorir(row):
                cores = {"Alto": "#ffd6d6", "Medio": "#fff4d6", "Baixo": "#d6f5d6"}
                c = cores.get(row["Risco"], "white")
                return [f"background-color: {c}"] * len(row)

            st.dataframe(
                exibir.style.apply(_colorir, axis=1),
                use_container_width=True,
                height=380,
            )

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=resultado["timestamp"],
                y=resultado["prob_pct"],
                mode="lines+markers",
                line=dict(color="#1f77b4", width=1.5),
                marker=dict(size=4),
                name="Probabilidade (%)",
            ))
            fig.add_hline(
                y=threshold * 100,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Threshold ({threshold*100:.2f}%)",
                annotation_position="top right",
            )
            fig.update_layout(
                xaxis_title="Data",
                yaxis_title="Probabilidade (%)",
                height=320,
                margin=dict(t=20, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

            fig_imp = _importancias_top5(modelo)
            if fig_imp:
                st.plotly_chart(fig_imp, use_container_width=True)

            csv_out = resultado[["timestamp", "prob_pct", "risco"]].copy()
            csv_out.columns = ["timestamp", "probabilidade_pct", "risco"]
            csv_out["timestamp"] = csv_out["timestamp"].dt.strftime("%Y-%m-%d")
            st.download_button(
                "Baixar resultados CSV",
                data=csv_out.to_csv(index=False, encoding="utf-8"),
                file_name=f"resultado_{horizonte}.csv",
                mime="text/csv",
            )


with aba_manual:
    st.markdown(
        f"Preencha os valores climaticos de **um unico dia** para obter a previsao "
        f"de risco com **{HORIZON_LABELS[horizonte]}** de antecedencia."
    )

    _cfg = {
        "PRECIP_DIARIA":  dict(min_value=0.0,    max_value=300.0,   value=0.0,    step=0.1,   label="Precipitacao diaria (mm)"),
        "TMEAN":          dict(min_value=-5.0,    max_value=45.0,    value=20.0,   step=0.1,   label="Temperatura media (graus C)"),
        "TMAX":           dict(min_value=-5.0,    max_value=50.0,    value=26.0,   step=0.1,   label="Temperatura maxima (graus C)"),
        "TMIN":           dict(min_value=-5.0,    max_value=40.0,    value=15.0,   step=0.1,   label="Temperatura minima (graus C)"),
        "URMEAN":         dict(min_value=0.0,     max_value=100.0,   value=70.0,   step=1.0,   label="Umidade relativa media (%)"),
        "RAD_SUM":        dict(min_value=0.0,     max_value=35000.0, value=14000.0,step=100.0, label="Radiacao global (kJ/m2)"),
        "PRESSAO_MED":    dict(min_value=880.0,   max_value=1030.0,  value=940.0,  step=0.5,   label="Pressao atmosferica (mB)"),
        "VENTO_MED":      dict(min_value=0.0,     max_value=20.0,    value=2.0,    step=0.1,   label="Velocidade do vento (m/s)"),
        "RAJADA_MAX":     dict(min_value=0.0,     max_value=50.0,    value=5.0,    step=0.1,   label="Rajada maxima (m/s)"),
        "P7":             dict(min_value=0.0,     max_value=600.0,   value=10.0,   step=1.0,   label="Precipitacao acumulada 7 dias (mm)"),
        "P15":            dict(min_value=0.0,     max_value=900.0,   value=20.0,   step=1.0,   label="Precipitacao acumulada 15 dias (mm)"),
        "P30":            dict(min_value=0.0,     max_value=1200.0,  value=40.0,   step=1.0,   label="Precipitacao acumulada 30 dias (mm)"),
        "P90":            dict(min_value=0.0,     max_value=2500.0,  value=150.0,  step=5.0,   label="Precipitacao acumulada 90 dias (mm)"),
        "DRY_STREAK_CUR": dict(min_value=0,       max_value=200,     value=0,      step=1,     label="Sequencia seca atual (dias)"),
        "DRY30_MAX":      dict(min_value=0,       max_value=30,      value=0,      step=1,     label="Maior seca em 30 dias (dias)"),
        "DRY90_MAX":      dict(min_value=0,       max_value=90,      value=0,      step=1,     label="Maior seca em 90 dias (dias)"),
        "TMEAN_MA7":      dict(min_value=-5.0,    max_value=45.0,    value=20.0,   step=0.1,   label="Temperatura media MA-7 (graus C)"),
        "TMEAN_MA30":     dict(min_value=-5.0,    max_value=45.0,    value=20.0,   step=0.1,   label="Temperatura media MA-30 (graus C)"),
        "URMEAN_MA7":     dict(min_value=0.0,     max_value=100.0,   value=70.0,   step=1.0,   label="Umidade relativa MA-7 (%)"),
        "URMEAN_MA30":    dict(min_value=0.0,     max_value=100.0,   value=70.0,   step=1.0,   label="Umidade relativa MA-30 (%)"),
        "PRESSAO_MED_MA7":dict(min_value=880.0,   max_value=1030.0,  value=940.0,  step=0.5,   label="Pressao MA-7 (mB)"),
        "SPI30_APRX":     dict(min_value=-4.0,    max_value=4.0,     value=0.0,    step=0.1,   label="SPI-30 aproximado (z-score)"),
    }

    with st.form("entrada_manual"):
        col_a, col_b, col_c = st.columns(3)
        cols_form = [col_a, col_b, col_c]
        valores = {}
        for idx, feat in enumerate(FEATURES):
            cfg = _cfg[feat]
            label = cfg.pop("label")
            col = cols_form[idx % 3]
            is_int = isinstance(cfg.get("step"), int)
            if is_int:
                valores[feat] = col.number_input(label, key=f"m_{feat}", **cfg)
            else:
                valores[feat] = col.number_input(label, key=f"m_{feat}", format="%.2f", **cfg)
            cfg["label"] = label

        submitted = st.form_submit_button("Calcular previsao", use_container_width=True)

    if submitted:
        X_manual = pd.DataFrame([valores])[FEATURES].fillna(0)
        prob = float(modelo.predict_proba(X_manual)[0, 1])
        st.divider()
        _exibir_resultado_unico(prob)
