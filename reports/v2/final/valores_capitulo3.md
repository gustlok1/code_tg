# Valores para o capitulo 3 (Metodologia)

Valores exatos que o capitulo de metodologia precisa citar. Todos vem de arquivos gerados pelo
pipeline. Para a ficha completa, ver `reports/v2/final/ficha_tecnica.md`.

## Testes automaticos
- Numero final de testes (pytest, `pipeline_v2/tests/`): 26, todos passando.
- Cobrem: causalidade das features, purga (nenhum treino com t+h dentro da validacao), calendario sem
  duplicatas, alvos iguais a vol(t+h), SPI/SPEI e anomalias ajustados so no treino, escalonador do LSTM
  so no treino, janela do LSTM termina em t, reconstrucao B3 + residual, baselines B1/B2/B3, gerador
  sintetico reprodutivel, conservacao de massa do balanco, gemeo dentro de 10% do real, schema igual
  ao real, e todo caminho de saida sintetico contendo "sintetico".

## Pontos da bacia (clima = media entre os pontos)
Quatro reservatorios do Sistema Cantareira:
- Jaguari-Jacarei: lat -22,9246; lon -46,4270; Vargem/SP; capacidade util 808,04 hm3.
- Cachoeira: lat -23,0511; lon -46,3200; Piracaia/SP; 69,65 hm3.
- Atibainha: lat -23,1757; lon -46,3936; Nazare Paulista/SP; 96,26 hm3.
- Paiva Castro: lat -23,3300; lon -46,6795; Franco da Rocha/SP; 7,61 hm3.
- Centroide: -23,1204; -46,4550. Capacidade do sistema: 981,56 hm3.

## Split e validacao
- Teste intocavel: 2023-01-01 em diante (aberto uma unica vez).
- Validacao: janela crescente, 4 folds, validando em 2003-2006, 2007-2012, 2013-2017 e 2018-2022.
- Purga: descarta no treino toda linha com t+h maior ou igual ao inicio do bloco de validacao.
- Seed global: 42.

## Grades de hiperparametros e valores escolhidos
### Regressao direta (Exp 2)
- Ridge: alpha em {1; 10}. RandomForestReg: n_estimators 200, max_depth em {None; 10}.
  XGBReg: n_estimators em {200; 300}, max_depth em {4; 6}, learning_rate 0,05.
### Classificacao (Exp 2)
- LogReg: C em {0,1; 1}. RandomForestClf: n_estimators 300, max_depth em {None; 8}.
  XGBClf: n_estimators em {300; 400}, max_depth em {4; 6}, lr 0,05, scale_pos_weight = neg/pos.
### Residual sobre o B3 (Exp 3) e modelo CONGELADO
- RidgeRes: alpha em {10; 100}.
- XGBRes: max_depth em {2; 3}, n_estimators 400, lr 0,03, reg_lambda em {10; 20}, subsample 0,8,
  colsample_bytree 0,8, min_child_weight 5.
- Escolhido por horizonte (congelamento.json):
  - h30: max_depth 3, n_estimators 400, lr 0,03, reg_lambda 20, subsample 0,8, colsample 0,8, min_child_weight 5.
  - h60: igual a h30.
  - h90: max_depth 2, n_estimators 400, lr 0,03, reg_lambda 10, subsample 0,8, colsample 0,8, min_child_weight 5.
- Thresholds de alerta congelados (alerta = volume previsto abaixo do valor):
  - limiar 40: h30 40,0 (B3 e XGBRes); h60 41,0 e 41,0; h90 43,0 (B3) e 41,0 (XGBRes).
  - limiar 30: h30 30,0 (B3) e 29,0 (XGBRes); h60 27,0 e 27,0; h90 27,0 e 27,0.

## LSTM
- Entrada: janelas de 180 dias das features de CLIMA_ESTADO (sem SPI), escalonadas (fit no treino do fold).
- 1 camada LSTM (hidden 32), dropout 0,2, Linear(32, 3) (uma saida por horizonte).
- Alvo: o residual sobre o B3. Early stopping nos 20% finais do treino (paciencia 8, max 60 epocas,
  Adam lr 1e-3, weight_decay 1e-4, batch 64). 3 seeds, media das previsoes. Passo de 5 dias entre
  janelas no treino. Orcamento de tempo 1200 s; execucao real 409 s.

## Deteccao de episodios
- Trechos continuos com volume abaixo do limiar, juntando lacunas menores que 60 dias e exigindo
  pelo menos 30 dias de duracao. Limiares 40 e 30.

## Metricas
- MAE, RMSE. skill contra baseline = 1 - MAE_modelo / MAE_baseline. IC95% por bootstrap em blocos de
  90 dias (2000 reamostragens). PR-AUC (entrada = so dias com volume acima do limiar), ROC-AUC, Brier.
- Antecedencia: dias entre o primeiro alerta nos 120 dias antes do inicio e o inicio do episodio.
- Falso-alarme por episodio: sequencia continua de alerta nao seguida de crise em ate h+30 dias.

## Versoes de software
- Python 3.13 (reproducao). Backend matplotlib Agg.
- Fixadas no requirements.txt: pandas 2.3.3, numpy 2.0.2, scikit-learn 1.6.1, xgboost 2.0.3,
  scipy 1.18.1, matplotlib 3.9.4, pyarrow 21.0.0, openpyxl 3.1.5, lxml 6.1.3, PyYAML 6.0.3,
  pytest 9.1.1, torch 2.14.1 (CPU), streamlit 1.50.0, plotly 6.7.0, joblib 1.5.3, seaborn 0.13.2.
- Runtime desta reproducao (mais novas que as fixadas): pandas 3.0.6, numpy 2.5.3, scikit-learn 1.9.1,
  xgboost 3.4.1, scipy 1.18.1, torch 2.14.1+cpu, matplotlib 3.11.2, lxml 6.1.3, pyarrow 25.0.1.
  Nota: os resultados sao da versao de runtime; para reproducao exata, recriar do requirements.txt.

## Rastreabilidade (hashes dos commits, branch tg2-v2)
- Congelamento (conjunto, modelos, hiperparametros e thresholds): f65b9aaa.
- Abertura do teste (uma unica vez): 0e89bd50.
- Diagnosticos pos-hoc do teste: 5254a394.
