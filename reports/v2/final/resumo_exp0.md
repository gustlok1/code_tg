# Exp 0 — Diagnóstico da v1

Objetivo: mostrar, com os próprios dados da v1, por que o resultado anterior falhou. Base do
`RELATORIO_ESTADO_ATUAL.md`. Código: `pipeline_v2/exp0_diagnostico_v1.py`. Figuras em `reports/v2/exp0/`.

## Achados (todos com número vindo dos dados)
1. Estações empilhadas e chuva somada (`chuva_vs_estacoes.png`): a "chuva anual" vai de 3.438 mm
   (2003, ~3,9 estações) a 50.476 mm (2019, ~41,5 estações). Cresce com o número de estações, não com
   o clima. O INMET foi agregado por soma, sem identificador de estação.
2. RAD_SUM zerada no teste (`rad_sum_zerada_teste.png`): por troca de grafia da coluna de radiação,
   RAD_SUM fica 0 em todo o período de teste (2020-2024) e inflada no treino (média 2019 = 677.231).
3. Validação cruzada sem positivos (`folds_sem_positivo.png`): no TimeSeriesSplit, os positivos por
   fold são [0, 0, 0, 15, 0] — 4 de 5 folds sem nenhum positivo. O threshold degenera e o F1 do CV é 0.
4. Regra simples bate os modelos (`spi_vs_modelos.png`): uma regra de limiar sobre o SPI30 tem
   AUC-ROC 0,785 / 0,755 / 0,752 (y30/60/90), acima de todos os modelos da v1 (melhor XGBoost 0,700 /
   0,418 / 0,592). O ML não agregava valor sobre uma regra trivial.

## Interpretação
A v1 media um artefato de agregação, não o clima; a avaliação estava quebrada (folds sem positivo) e
o modelo não superava uma regra simples. Isso motivou a reconstrução: alvo objetivo (volume do SAR),
sem agregação indevida, validação com purga e baselines honestos.
