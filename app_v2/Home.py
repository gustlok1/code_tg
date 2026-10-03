# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

st.set_page_config(page_title="Crise hídrica no Cantareira", layout="wide")

paginas = [
    st.Page("paginas/situacao_atual.py", title="Situação atual",
            url_path="situacao-atual", default=True),
    st.Page("paginas/retrospectiva.py", title="Retrospectiva das crises",
            url_path="retrospectiva"),
    st.Page("paginas/experimento_sintetico.py", title="Experimento sintético",
            url_path="experimento-sintetico"),
    st.Page("paginas/sobre.py", title="Sobre", url_path="sobre"),
]
st.navigation(paginas).run()
