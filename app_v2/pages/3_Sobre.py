# -*- coding: utf-8 -*-
import streamlit as st

st.set_page_config(page_title="Sobre", layout="wide")
st.title("Sobre o projeto")
st.markdown(
    """
Trabalho de Graduação do curso de Ciência de Dados da Fatec Santana de Parnaíba.
Autores: Victor Ribeiro Cunha e Gustavo Henrique Moises Martins. Orientador: Prof. Dr. Inacio Henrique Yano.

## O que este app mostra
- Volume útil atual do Sistema Cantareira (SAR/ANA) e previsão em 30, 60 e 90 dias, com faixa de
  incerteza e a faixa operacional ANA/DAEE prevista.
- Backtest das crises conhecidas (2003-04, 2013-16, 2021-22, 2025-26).
- O experimento sintético, sempre rotulado como dado sintético.

## Metodologia (resumo)
- Alvo objetivo: volume útil do sistema. Crise pela Resolução ANA/DAEE 925/2017 (abaixo de 40% é
  Alerta; abaixo de 30% é grave).
- Clima: ERA5-Land (Open-Meteo) e NASA POWER, média entre os 4 reservatórios do sistema
  (Jaguari-Jacareí, Cachoeira, Atibainha e Paiva Castro).
- Modelo: XGBoost que prevê o resíduo sobre um baseline forte (persistência + variação sazonal, B3).
  Validação com janela crescente e purga; teste de 2023 em diante aberto uma única vez, após o
  congelamento das escolhas.
- Detalhes completos em reports_v2/final/ficha_tecnica.md.

## Resultado, em uma frase
O estado do reservatório é o preditor decisivo; o clima agrega pouco na validação (empata com o B3),
mas no teste recente o modelo superou o baseline e antecipou a crise de 2025-26.
"""
)
