# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
SINT = ROOT / "reports_v2" / "sintetico"

st.set_page_config(page_title="Sintético", layout="wide")
st.markdown("<div style='background:#7f0000;color:white;padding:8px 14px;border-radius:6px;"
            "font-weight:bold;display:inline-block'>DADOS SINTÉTICOS — não são dados reais</div>",
            unsafe_allow_html=True)
st.title("Experimento sintético (Exp 1)")
st.caption("Ambiente controlado para provar que o pipeline recupera um sinal conhecido e para medir "
           "onde o ML supera o B3. Nada aqui é dado real do Cantareira.")

st.markdown(
    "- O skill do ML sobre o B3 cresce com a persistência efetiva da chuva (φ_ef) e com o número de "
    "crises N.\n"
    "- Com o alvo embaralhado o skill vai a zero; com a chuva futura (oráculo) o skill dispara: o "
    "limite é a informação, não o algoritmo.\n"
    "- O Cantareira real está em φ ≈ 0,013 (chuva sem persistência mensal)."
)

figs = [("skill_vs_phief.png", "Skill vs B3 (h=90) por persistência efetiva φ_ef (gêmeo destacado)"),
        ("heatmap_skill_phi_N.png", "Skill vs B3 em φ × N"),
        ("barras_sanidade.png", "Sanidade: real vs embaralhado vs oráculo"),
        ("serie_exemplo.png", "Série sintética de exemplo (chuva, afluência, volume)")]
for arq, leg in figs:
    p = SINT / arq
    if p.exists():
        st.image(str(p), caption=f"[DADOS SINTÉTICOS] {leg}", use_container_width=True)

cal = SINT / "calibracao_real.csv"
if cal.exists():
    st.subheader("Calibração com o real")
    st.dataframe(pd.read_csv(cal), hide_index=True)
