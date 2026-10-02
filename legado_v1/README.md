# legado_v1 — versao do TG1, congelada

Versao 1 do trabalho (TG1), mantida apenas como referencia e base do Exp 0 (o diagnostico de por que a
v1 falhou, em docs/RELATORIO_ESTADO_ATUAL.md). Nao e desenvolvida nem corrigida. Os defeitos conhecidos
(estacoes do INMET empilhadas sem identificador e com chuva somada, RAD_SUM zerada no teste, validacao
cruzada sem positivos) sao justamente o que o Exp 0 mostra.

## Estrutura
- `pipeline/`: scripts da v1 (01_clean_hourly.py, 02_build_features.py, 03_build_labels.py,
  04_train_model.py).
- `app/`: aplicacao Streamlit da v1.
- `models/`: modelos .pkl treinados na v1.
- `reports/threshold_padrao/` e `reports/threshold_otimizado/`: relatorios da v1 (comparativos de
  modelos, matrizes de confusao, importancia de features). O Exp 0 le os comparativos de
  `reports/threshold_otimizado/`.
- `data/`: dados da v1 (fora do git, ver abaixo).

## Dados (fora do git)
`legado_v1/data/` esta no .gitignore porque inclui arquivos pesados (acima de 5 MB): o horario limpo
`data/interim/inmet_sp_hourly_clean.parquet` (cerca de 97 MB) e os brutos `data/raw/*.xlsx` (cerca de
439 MB no total). O historico do git ainda contem versoes antigas desses arquivos (sem reescrita); eles
apenas deixaram de ser rastreados.

- Fonte: INMET, Banco de Dados Meteorologicos (BDMEP), https://bdmep.inmet.gov.br/. Estacoes
  automaticas do estado de Sao Paulo, dados horarios.
- Periodo: 2003 a 2024.
- Arquivos esperados por quem for rodar o Exp 0:
  - `legado_v1/data/raw/AAAA_base.xlsx` (um por ano, 2003 a 2024).
  - `legado_v1/data/interim/inmet_sp_hourly_clean.parquet` e `qc_hourly.csv`.
  - `legado_v1/data/features/inmet_sp_daily.parquet`, `_features.parquet`, `_labels.parquet`.
  - `legado_v1/data/events/eventos.csv`.

## Como regenerar
A partir dos .xlsx brutos em `legado_v1/data/raw/`, rode os scripts da v1 na ordem (ajustando os
caminhos de entrada/saida para dentro de `legado_v1/data/`):

```
python legado_v1/pipeline/01_clean_hourly.py   # raw -> interim (inmet_sp_hourly_clean.parquet + qc)
python legado_v1/pipeline/02_build_features.py  # interim -> features diarias
python legado_v1/pipeline/03_build_labels.py    # features + eventos -> rotulos y30/y60/y90
```

O Exp 0 (pipeline_v2/exp0_diagnostico_v1.py) le apenas as saidas ja existentes em `legado_v1/data/` e
`legado_v1/reports/threshold_otimizado/`; nao recalcula a v1.
