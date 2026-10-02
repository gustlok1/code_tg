# Ficha técnica (TG2 v2) — tudo que a metodologia do texto precisa citar

Documento de referência gerado para o texto. Todo número vem de arquivo produzido pelo
pipeline (ver caminhos). Hashes de congelamento e de abertura do teste no fim.

## 1. Pergunta, alvo e crise
- Pergunta: variáveis climáticas, sozinhas ou com o estado do reservatório, antecipam em 30, 60 e 90
  dias a entrada do Sistema Cantareira em faixa de crise?
- Alvo: volume útil do sistema (%) em t+30, t+60 e t+90 (fonte SAR/ANA).
- Crise (Resolução Conjunta ANA/DAEE 925/2017): principal volume < 40% (17 episódios 1984-2026);
  grave < 30% (5 episódios: 1986, 2003-04, 2013-16, 2021-22, 2025-26).

## 2. Pontos da bacia (clima = média diária entre os pontos)
4 reservatórios do Sistema Cantareira (`config.yaml`):

| Reservatório | lat | lon | município | capacidade útil (hm³) |
|---|---|---|---|---|
| Jaguari-Jacareí | -22,9246 | -46,4270 | Vargem/SP | 808,04 |
| Cachoeira | -23,0511 | -46,3200 | Piracaia/SP | 69,65 |
| Atibainha | -23,1757 | -46,3936 | Nazaré Paulista/SP | 96,26 |
| Paiva Castro | -23,3300 | -46,6795 | Franco da Rocha/SP | 7,61 |

Centroide: -23,1204, -46,4550. Capacidade do sistema (reconstruída): 981,56 hm³ (4 reservatórios).

## 3. Fontes de dados (`data/raw_v2/_manifest.json`)
- SAR/ANA (`/sar0/MedicaoCantareira`): volume útil (hm³ e %), afluência e defluência diários,
  1984-01-01 a 2026-10-01 (15.615 dias). Volume negativo (reserva técnica) mantido; mínimo do
  sistema -23,21% em 2015-02-02.
- ERA5-Land via Open-Meteo (`archive-api.open-meteo.com`): chuva, ET0, temperatura (média/máx/mín),
  radiação, 1984-2026, nos 4 pontos.
- NASA POWER (`power.larc.nasa.gov`): chuva, temperatura, radiação, umidade, vento, pressão, 1984-2026.
  Usado só no teste de robustez (anomalia de chuva em 1999; ERA5 é melhor).
- Dataset diário final: `data/processed_v2/dataset_diario.parquet`, 15.616 dias (1984-01-01 a
  2026-10-02), 72 colunas, 0,4% de células ausentes.

## 4. Conjuntos de features (nome, janela, fonte, causal)
Todas causais (só t e passado). SPI/SPEI/anomalias ajustados no treino de cada fold.

### CLIMA (clima ERA5 + SPI/SPEI)
- chuva acumulada: `era5_precip_acc{7,30,90,180,365}` (mm, janela 7/30/90/180/365 d, ERA5).
- ET0 acumulada: `et0_acc{7,30,90,180,365}` (mm, ERA5).
- balanço: `era5_bal_acc{7,30,90,180,365}` = chuva − ET0 (mm, ERA5).
- temperatura: `era5_tmean_ma{7,30}` (°C, média móvel 7/30 d, ERA5).
- sazonalidade: `doy_sin`, `doy_cos` (dia do ano, calendário).
- SPI: `spi_{3,6,12}` (meses; chuva; gama por mês, fit no treino).
- SPEI: `spei_{3,6,12}` (meses; balanço P−ET0; gama deslocada, fit no treino).

### CLIMA_ESTADO = CLIMA + estado do reservatório
- `vol_pct` (volume útil em t, %), `vol_var7`, `vol_var30` (variação em 7/30 d, p.p.),
  `aflu_ma7`, `aflu_ma30`, `aflu_ma90` (afluência média, m³/s),
  `deflu_ma30` (defluência média, m³/s), `pos_2017` (0/1, vigência da Resolução 925 desde 2017-05-29).

### Residual (Exp 3) = CLIMA_ESTADO + anomalias sazonais
- `anom_era5_precip_acc{90,180,365}`: anomalia da chuva acumulada contra a climatologia por dia do
  ano, ajustada no treino (`transformadores.anomalias_sazonais`).
- Alvo do residual: r_h = vol(t+h) − B3(t); previsão final = B3(t) + r_h previsto.

## 5. Split e validação
- Teste intocável: 2023-01-01 em diante (aberto uma única vez pelo `03`).
- Validação: janela crescente, 4 folds, validando em 2003-2006, 2007-2012, 2013-2017, 2018-2022,
  treinando com tudo que vem antes.
- Purga: no treino de cada fold descarta-se toda linha com t+h ≥ início do bloco de validação (o alvo
  olha h dias à frente). O mesmo antes do teste.

## 6. Baselines (régua = skill contra B3)
- B1 persistência: previsão = volume em t.
- B2 climatologia: média do volume por dia do ano, ajustada no treino, avaliada no dia do ano de t+h.
- B3 persistência + variação sazonal: volume em t + variação média em h dias para aquele dia do ano,
  ajustada no treino. É a régua.
- Classificação (baselines): crise quando a previsão fica abaixo do limiar; mais a regra SPI-6 < −1.

## 7. Grades de hiperparâmetros e valores escolhidos
Seleção pela média dos 4 folds de validação.

### Regressão direta (Exp 2, `modelagem_v2.GRADE_REG`)
- Ridge: alpha ∈ {1, 10} (com StandardScaler).
- RandomForestReg: n_estimators 200, max_depth ∈ {None, 10}.
- XGBReg: n_estimators ∈ {200, 300}, max_depth ∈ {4, 6}, learning_rate 0,05.

### Classificação (Exp 2, `GRADE_CLF`)
- LogReg: C ∈ {0,1; 1} (StandardScaler, class_weight balanced).
- RandomForestClf: n_estimators 300, max_depth ∈ {None, 8} (class_weight balanced).
- XGBClf: n_estimators ∈ {300, 400}, max_depth ∈ {4, 6}, lr 0,05, scale_pos_weight = neg/pos.

### Residual sobre B3 (Exp 3, `analise_exp3.GRADE_RES`) — modelo congelado
- RidgeRes: alpha ∈ {10, 100}.
- XGBRes: {max_depth ∈ {2,3}, n_estimators 400, lr 0,03, reg_lambda ∈ {10,20}, subsample 0,8,
  colsample_bytree 0,8, min_child_weight 5}.
- Valores CONGELADOS (XGBRes, `reports/v2/final/congelamento.json`):
  - h30: max_depth 3, n_estimators 400, lr 0,03, reg_lambda 20, subsample 0,8, colsample 0,8, min_child_weight 5.
  - h60: igual a h30.
  - h90: max_depth 2, n_estimators 400, lr 0,03, reg_lambda 10, subsample 0,8, colsample 0,8, min_child_weight 5.
- Thresholds de alerta CONGELADOS (volume previsto abaixo do valor = alerta), por limiar/horizonte/modelo:
  - limiar 40: h30 B3/XGB 40,0/40,0; h60 41,0/41,0; h90 43,0/41,0.
  - limiar 30: h30 B3/XGB 30,0/29,0; h60 27,0/27,0; h90 27,0/27,0.
- seed global 42.

## 8. LSTM (`lstm_v2.py`)
- Entrada: janelas dos últimos 180 dias das features de CLIMA_ESTADO (sem SPI), escalonadas
  (StandardScaler fit no treino do fold; a janela termina em t).
- Arquitetura: 1 camada LSTM (hidden 32), dropout 0,2, Linear(32, 3) (uma saída por horizonte).
- Alvo: o resíduo sobre o B3 (r_30, r_60, r_90). Early stopping nos 20% finais cronológicos do treino
  (paciência 8, máx 60 épocas, Adam lr 1e-3, weight_decay 1e-4, batch 64). 3 seeds, média das previsões.
- Treino com passo (stride) de 5 dias entre janelas; orçamento de tempo (default 1200 s). Resultado:
  409 s, empate técnico com o XGBRes (skill_B3 médio 0,022 vs 0,021).

## 9. Transformadores (fit no treino, transform no resto)
- SPI: chuva acumulada em k meses; gama por mês do ano (floc=0); fração de zeros como massa pontual;
  z = Φ⁻¹(H). SPEI: igual sobre D = P−ET0, deslocado para o domínio positivo (offset do treino).
- Anomalia sazonal: valor − climatologia por dia do ano (média do treino).

## 10. Detecção de episódios (`dataset_v2.detectar_episodios`, `config.yaml`)
- Trechos contínuos com volume < limiar, juntando lacunas menores que 60 dias e exigindo pelo menos
  30 dias de duração. Limiares 30 e 40. Arquivos: `episodios_crise.csv` (30), `episodios_crise40.csv` (40).

## 11. Definição exata das métricas
- MAE = média de |real − previsto| (p.p. de volume). RMSE = raiz da média dos quadrados.
- skill contra baseline = 1 − MAE_modelo / MAE_baseline (>0 = melhor que o baseline).
- IC95% do skill: bootstrap em blocos de 90 dias consecutivos (2000 reamostragens).
- Classificação: PR-AUC (average precision), ROC-AUC, Brier (só modelos probabilísticos), prevalência.
  Recortes: (a) todos os dias; (b) ENTRADA = só dias com volume em t acima do limiar.
- Antecedência: dias entre o primeiro alerta nos 120 dias antes do início do episódio e o início.
  Alerta = volume previsto abaixo do threshold (threshold por F1 na validação).
- Falso-alarme por episódio: sequência contínua de alerta que NÃO é seguida de crise (volume real
  abaixo do limiar) em até h+30 dias. Reportado como contagem absoluta e taxa por ano.

## 12. Gerador sintético e gêmeo calibrado (`sintetico_v2.py`)
- Gerador (40 anos diários, seed por cenário): chuva = ocorrência Bernoulli sazonal × quantidade gama
  (climatologia por mês da ERA5 real) × exp(anomalia). Anomalia mensal AR(1) (persistência φ, desvio σ)
  + N secas injetadas (6-12 meses, fator 0,4-0,6). Afluência: reservatório linear sobre o escoamento
  (τ 30-60 d; runoff 0,22; área 2280 km²). Balanço: V(t+1) = V(t) + afluência − retirada − evaporação,
  capacidade 981,56 hm³, piso -30%, retirada reduzida pelas faixas 60/40/30/20%.
- Grade: φ ∈ {0; 0,5; 0,8; 0,95} × N ∈ {3,10,30} × 3 seeds, mais N=0. Persistência EFETIVA φ_ef =
  autocorrelação lag-1 das anomalias mensais (mesma função no real e no sintético). φ real = 0,0129.
- Gêmeo calibrado: τ = 24 d (ajuste à afluência real, erro_rel 0,63), ruído multiplicativo na
  afluência σ = 0,6266, σ da anomalia calibrado ao desvio mensal real = 0,25 (o σ_real do AR1 = 0,6944
  superdispersa), φ real, N=0, 10 seeds. Chuva mensal do gêmeo dentro de 10% do real (média -2,2%,
  desvio -3,6%). LIMITAÇÃO: o gêmeo ainda supera o B3 em h30 (+0,27) e h60 (+0,14), empata em h90,
  porque o volume sintético é previsível demais (não modela operação/transferências/múltiplos reservatórios).

## 13. Versões de software
- Python 3.13 (reprodução). Backend matplotlib Agg.
- Fixadas em `requirements.txt`: pandas 2.3.3, numpy 2.0.2, scikit-learn 1.6.1, xgboost 2.0.3,
  scipy 1.18.1, matplotlib 3.9.4, pyarrow 21.0.0, openpyxl 3.1.5, lxml 6.1.3, PyYAML 6.0.3,
  pytest 9.1.1, torch 2.14.1 (CPU), streamlit 1.50.0, plotly 6.7.0, joblib 1.5.3, seaborn 0.13.2.
- Versões de runtime desta reprodução (venv de trabalho, mais novas que as fixadas): pandas 3.0.6,
  numpy 2.5.3, scikit-learn 1.9.1, xgboost 3.4.1, scipy 1.18.1, torch 2.14.1+cpu, matplotlib 3.11.2,
  PyYAML 6.0.3, lxml 6.1.3, pyarrow 25.0.1. NOTA de rigor: os resultados foram produzidos com as
  versões de runtime; para reprodução exata do texto, recriar o ambiente a partir do requirements.txt.

## 14. Rastreabilidade (hashes dos commits)
- Congelamento (conjunto, modelos, hiperparâmetros e thresholds): commit `f65b9aaa`
  (arquivo `reports/v2/final/congelamento.json`).
- Abertura do teste (uma única vez): commit `0e89bd50` (registro em `reports/v2/final/teste_aberto.log`).
- Diagnósticos pós-hoc do teste: commit `5254a394` (`reports/v2/final/teste_diagnostico.csv`).
- Branch: `tg2-v2`.
