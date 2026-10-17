# Documentacao do TG2 — Previsao de crises hidricas no Sistema Cantareira

Ponto de entrada da documentacao. Daqui da para chegar em tudo: como rodar, o que cada
experimento mostra, os valores exatos para o texto, as figuras finais, as decisoes de projeto
e o diagnostico da versao 1. Atualizado em 17 de outubro de 2026.

Trabalho de Graduacao do curso de Ciencia de Dados da Fatec Santana de Parnaiba. Autores: Victor
Ribeiro Cunha e Gustavo Henrique Moises Martins. Orientador: Prof. Dr. Inacio Henrique Yano.

## Em uma frase

Prever, com 30, 60 e 90 dias de antecedencia, a entrada do Sistema Cantareira em faixa de crise
(volume util abaixo de 40% e Alerta; abaixo de 30% e grave, pela Resolucao Conjunta ANA/DAEE
925/2017), a partir de dados climaticos (ERA5-Land) e do estado dos reservatorios (SAR/ANA).

## Resultado principal

- O estado do reservatorio e o preditor decisivo. O clima agregado mais o modelo de ML (XGBoost
  sobre o residuo do baseline B3) empata com o baseline forte B3 (persistencia mais variacao
  sazonal) na validacao de 20 anos.
- No teste intocado (2023 em diante, modelo congelado), o modelo supera o B3 (skill de 0,167 a
  0,240) e antecipa a crise de 2025-26 em ate 11 dias.
- Resultado negativo e reportado como e. O teste foi aberto uma unica vez, apos o congelamento das
  escolhas (conjunto, hiperparametros e limiares).

## Mapa da documentacao

Comece por aqui | Para que serve
---|---
[../README.md](../README.md) | Visao geral, como instalar e como rodar o pipeline e o app.
[RELATORIO_ESTADO_ATUAL.md](RELATORIO_ESTADO_ATUAL.md) | Auditoria da versao 1: por que a v1 falhou (base do Exp 0).
[../DECISOES.md](../DECISOES.md) | Diario de decisoes de projeto, com data, motivo e alternativa descartada.
[../CLAUDE.md](../CLAUDE.md) | Contexto, pergunta de pesquisa, hipoteses, backlog e regras de rigor.

### Texto do TG (capitulo 4 e metodologia)

Documento | Conteudo
---|---
[capitulo4_rascunho.md](../reports_v2/final/capitulo4_rascunho.md) | Rascunho do capitulo 4 (Analise e Resultados), secao por secao.
[ficha_tecnica.md](../reports_v2/final/ficha_tecnica.md) | Ficha tecnica: todo numero que a metodologia precisa citar, com a fonte.
[valores_capitulo3.md](../reports_v2/final/valores_capitulo3.md) | Valores exatos da metodologia (testes, pontos da bacia, grades, hashes).
[dicionario_de_dados.csv](../reports_v2/final/dicionario_de_dados.csv) | Dicionario de dados (cada variavel do dataset).

### Resultados por experimento

Experimento | Resumo
---|---
Exp 0 — diagnostico da v1 | [resumo_exp0.md](../reports_v2/final/resumo_exp0.md)
Exp 1 — sintetico (sinal conhecido) | [resumo_exp1.md](../reports_v2/final/resumo_exp1.md)
Exp 2 — clima contra clima e estado | [resumo_exp2.md](../reports_v2/final/resumo_exp2.md)
Exp 3 — modelos contra a regua B3 | [resumo_exp3.md](../reports_v2/final/resumo_exp3.md)
Exp 4 — outros reservatorios | [resumo_exp4.md](../reports_v2/final/resumo_exp4.md)

### Figuras, tabelas e verificacoes

Documento | Conteudo
---|---
[figuras/indice_figuras.md](../reports_v2/final/figuras/indice_figuras.md) | Indice das figuras finais (fig_00 a fig_16), com uma frase de leitura de cada.
[tabela_antecedencia_por_episodio.md](../reports_v2/final/tabela_antecedencia_por_episodio.md) | Antecedencia do alerta por episodio (XGBRes vs B3), validacao e teste.
[verificacoes_orientador.md](../reports_v2/final/verificacoes_orientador.md) | Respostas as verificacoes do orientador, com arquivo de origem de cada numero.
[reproducao.md](../reports_v2/final/reproducao.md) | Reproducao do congelamento, do teste e da PR-AUC de entrada (51/51 itens identicos).

### Codigo

Pasta | O que tem
---|---
[../pipeline_v2/README.md](../pipeline_v2/README.md) | Guia do pipeline da v2 (coleta, dataset, modelos, experimentos, testes).
`../app_v2/` | Aplicacao Streamlit (situacao atual, retrospectiva, sintetico, sobre).

## Estado atual e ultimas atualizacoes

O pipeline esta completo, com 29 testes automaticos verdes e reproducao bit a bit das escolhas
congeladas, do teste e da PR-AUC de entrada (51 de 51 itens identicos, ver `reproducao.md`).
As rodadas mais recentes (outubro de 2026), em detalhe no `DECISOES.md`:

- Verificacoes e figuras para o orientador: novo mapa da area de estudo (`fig_00`), tabela de
  antecedencia por episodio e respostas 1a a 1g com a fonte de cada numero. Achado reportado: o
  ganho do sintetico cresce com a persistencia da chuva, nao de forma monotonica com o numero de
  secas.
- Gerador reproduzivel da PR-AUC de entrada (`02e_prauc_entrada_ic.py`), fechando a ressalva de
  reprodutibilidade: o arquivo historico e reproduzido bit a bit.
- Ajustes de apresentacao das figuras (persistencia efetiva em notacao matematica, selos de dado
  sintetico, anotacoes e eixos).
- App: a faixa de incerteza da "Situacao atual" passou a vir do erro fora da amostra da validacao
  (percentil 80, 1984 a 2022), nao mais dos residuos de treino.

## Dados (fora do git)

Os dados brutos e processados (`data/raw_v2/`, `data/processed_v2/`, `data/sintetico/`) ficam fora
do controle de versao por serem pesados e reconstrutiveis. Regenere-os com
`pipeline_v2/00_download_dados.py --all` e `pipeline_v2/01_build_dataset_v2.py`. O Word do TG
(`docs/TG2_Previsao_Crises_Hidricas.docx`) tambem fica fora do git (ver `.gitignore`).
