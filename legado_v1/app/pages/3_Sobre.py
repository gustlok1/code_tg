import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from utils import carregar_modelo, FEATURES, CRISES

st.set_page_config(page_title="Sobre", page_icon=None, layout="wide")
st.title("Sobre o Projeto")

modelo, threshold = carregar_modelo()

st.markdown(
    """
## Sistema de Previsao de Crise Hidrica

Este sistema foi desenvolvido como **Trabalho de Graduacao** do curso de
**Tecnologia em Ciencia de Dados** da **Fatec Santana de Parnaiba**,
com o objetivo de antecipar crises hidricas no **Sistema Cantareira** de Sao Paulo
usando dados climaticos e tecnicas de Machine Learning.
"""
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Metodologia")
    st.markdown(
        f"""
**Pipeline de dados:**
1. Coleta de dados horarios do INMET (22 arquivos xlsx, 2003 a 2024)
2. Limpeza e unificacao em Parquet (aproximadamente 5,5 milhoes de registros horarios)
3. Agregacao diaria e criacao de 22 features climaticas
4. Rotulagem supervisionada: dias com crise hidrica iminente (`y90`, `y60`, `y30`)

**Rotulagem:**
- `y90 = 1` para os 15 dias imediatamente anteriores ao inicio de cada crise
- Dias durante a crise sao excluidos do treino (`NaN`)
- 3 eventos historicos rotulados, totalizando **45 dias positivos** em 8.036 dias totais

**Treinamento:**
- Split temporal 80/20 sem shuffle (sem vazamento de dados)
- Treino: 2003 a 2020 / Teste: 2021 a 2024
- Validacao cruzada temporal com `TimeSeriesSplit` (5 folds)
- Metrica principal: **F1-score** + **AUC-ROC**

**Threshold:** o limiar de classificacao foi otimizado por busca em grade
(0,01 a 0,99) via validacao cruzada, resultando em threshold = `{threshold:.2f}`.
"""
    )

with col2:
    st.subheader("Dados e Fontes")
    st.markdown(
        """
**Fonte dos dados climaticos:**
[INMET - Instituto Nacional de Meteorologia](https://bdmep.inmet.gov.br/)
Estacao: Sao Paulo / Mirante (A701)
Cobertura: 2003 a 2024 (dados horarios)

**Eventos de crise rotulados:**
"""
    )
    for c in CRISES:
        st.markdown(f"- **{c['label']}**: {c['start']} a {c['end']}")

    st.markdown(
        """
**Features climaticas utilizadas (22):**
Precipitacao diaria e acumulada (P7/P15/P30/P90),
temperatura (media, maxima, minima, medias moveis),
umidade relativa (media e medias moveis),
radiacao global, pressao atmosferica,
velocidade e rajada de vento,
sequencias secas (dry spells),
SPI-30 aproximado (z-score mensal de P30).
"""
    )

st.divider()

st.subheader("Modelos comparados")
st.markdown(
    f"""
| Algoritmo | Parametros principais | Tratamento de desbalanceamento |
|-----------|----------------------|-------------------------------|
| Decision Tree | `max_depth=10` | `class_weight='balanced'` |
| **Random Forest** | `n_estimators=200, max_depth=15` | `class_weight='balanced'` |
| XGBoost | `n_estimators=200, max_depth=8, lr=0.1` | `scale_pos_weight aprox. 186` |

O dataset apresenta **forte desbalanceamento**: aproximadamente 0,64% de amostras positivas
(45 dias de pre-crise em cerca de 7.002 dias de treino e teste).

O modelo em producao neste aplicativo e: **{type(modelo).__name__}**
com threshold otimizado = `{threshold:.2f}`.
"""
)

st.divider()

st.subheader("Limitacoes")
st.markdown(
    """
- O dataset cobre apenas **3 eventos historicos** de crise hidrica, limitando a
  capacidade de generalizacao do modelo.
- As features sao estritamente climaticas; variaveis hidrologicas (nivel do reservatorio,
  volume util) nao foram incluidas nesta versao.
- A previsao e valida para o Sistema Cantareira e para o contexto climatico de Sao Paulo;
  nao generaliza para outras regioes sem re-treino.
- O modelo prevê risco de **pre-crise** (dias que antecedem o inicio de um evento);
  periodos durante uma crise ativa recebem probabilidade proxima de zero, pois o modelo
  foi treinado para detectar a janela de antecedencia, nao o estado de crise em si.
- O threshold otimizado foi ajustado para maximizar F1 no conjunto de validacao,
  mas em producao pode ser necessario ajusta-lo conforme a tolerancia ao alarme falso desejada.
"""
)
