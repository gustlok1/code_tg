# Verificações para a versão do orientador

Rodada de conferência (somente leitura dos arquivos já gerados). Nenhum número de
resultado foi recalculado ou alterado; o teste não foi reaberto. Cada resposta cita
o arquivo de origem.

## 1a. Como o limiar de alerta sobre o volume previsto é escolhido (B3 e XGBRes)

Fontes: `pipeline_v2/analise_exp3.py` (função `_threshold_vol`), `pipeline_v2/03_congelar_e_testar.py`,
`reports_v2/final/congelamento.json`.

O alerta dispara quando o volume PREVISTO fica abaixo de um limiar. Esse limiar é
escolhido separadamente para cada modelo (B3 e XGBRes), cada horizonte e cada faixa
de crise (40 e 30), varrendo os valores inteiros de 20 a 59 e ficando com o que
maximiza o F1 nas previsões out-of-fold da validação (rótulo positivo: volume real
abaixo da faixa). Os limiares escolhidos são congelados em `congelamento.json` e
aplicados FIXOS ao teste, sem reescolha (o 03 lê `thresholds_volume_alerta`). Ou seja,
na validação e no teste valem os mesmos limiares.

Valores por horizonte (h30 / h60 / h90), confirmados em `antecedencia_teste.csv` e
`congelamento.json`:

- Faixa 40: B3 = 40 / 41 / 43 ; XGBRes = 40 / 41 / 41. (bate com o que você anotou)
- Faixa 30: B3 = 30 / 27 / 27 ; XGBRes = 29 / 27 / 27.

## 1b. Episódios de alarme falso por horizonte (faixa 40), XGBRes e B3

Validação, fonte `reports_v2/resultados/antecedencia_pareada.csv`
(colunas `falso_alarme_abs_*` e `falso_alarme_ano_*`):

| Horizonte | XGBRes (abs) | XGBRes (por ano) | B3 (abs) | B3 (por ano) |
|---|---|---|---|---|
| 30 | 2 | 0,10 | 1 | 0,05 |
| 60 | 3 | 0,15 | 0 | 0,00 |
| 90 | 4 | 0,20 | 0 | 0,00 |

Teste, fonte `reports_v2/final/antecedencia_teste.csv` (coluna `falso_alarme_abs_teste`):

| Horizonte | XGBRes (abs) | B3 (abs) |
|---|---|---|
| 30 | 0 | 0 |
| 60 | 2 | 0 |
| 90 | 0 | 0 |

Leitura: o B3 praticamente não dá alarme falso; o custo do XGBRes é de poucos episódios
de alarme falso (3 a 4 em 20 anos na validação, 2 em h60 no teste).

## 1c. AUC-ROC, Brier e RMSE: existem em CSV?

Resumo: os três existem na VALIDAÇÃO; nenhum está no CSV final do teste (que traz só
MAE, skill e PR-AUC). Nada foi calculado de novo.

- RMSE: existe. Fonte `reports_v2/resultados/resumo_regressao.csv` (média dos folds).
  CLIMA_ESTADO, h30 / h60 / h90: B3 = 4,219 / 6,781 / 9,032 ; XGBReg = 4,409 / 6,820 / 9,372.
  No teste (`reports_v2/final/metricas_teste.csv`) não há RMSE, só MAE
  (B3 = 2,949 / 4,866 / 6,267 ; XGBRes = 2,242 / 3,750 / 5,221).
- ROC-AUC: existe. Fonte `reports_v2/resultados/resumo_classificacao.csv` (e por fold em
  `metricas_validacao.csv`). Está na tarefa de classificação; o classificador de ML é o
  XGBClf (não o XGBRes). Exemplo CLIMA_ESTADO, faixa 40, recorte "entrada", h30:
  B3 = 0,9868 ; XGBClf = 0,9786. No teste não há ROC-AUC.
- Brier: existe só para os modelos probabilísticos (LogReg, RF, XGBClf) em
  `resumo_classificacao.csv`; B1 e B3 não têm Brier (não dão probabilidade). Exemplo
  XGBClf CLIMA_ESTADO faixa 40 entrada h30: Brier = 0,0274. No teste não há Brier.

## 1d. Origem das Tabelas 9 e 11 (PR-AUC do B3 a 30 dias: 0,748 vs 0,805)

Confirmado: uma é a média das rodadas, a outra é a agregação out-of-fold.

- 0,748 vem de `reports_v2/resultados/resumo_classificacao.csv`
  (CLIMA_ESTADO, B3, h30, faixa 40, recorte "entrada"). É a MÉDIA dos 4 folds
  (`02_treino_v2.py` faz `groupby(...).mean()`). Os PR-AUC por fold são
  0,9923 / 0,432 / 0,5808 / 0,9871, cuja média é 0,748. Prevalência (média dos folds) 0,0737.
- 0,805 vem de `reports_v2/resultados/prauc_entrada_ic.csv` (h30, B3 = 0,8047), com
  IC95% bootstrap [0,5835; 0,9827] e uma única prevalência agregada 0,0624. Pela estrutura
  (um valor por horizonte/modelo, com IC e prevalência única) é a PR-AUC AGREGADA
  out-of-fold (junta as previsões de todos os folds numa série só). A diferença de
  prevalência (0,0737 média dos folds contra 0,0624 agregada) é a própria assinatura de
  "média das rodadas" contra "out-of-fold agregado".
- Ressalva de reprodutibilidade: nenhum script atual de `pipeline_v2/` regenera
  `prauc_entrada_ic.csv` (o arquivo vem do commit de saneamento; a natureza acima é
  confirmada pela estrutura e pelos valores, não por uma reexecução). Vale decidir se esse
  CSV deve passar a ser gerado por um orquestrador, para o texto poder referenciá-lo com segurança.

## 1e. Paiva Castro no SAR e como o Exp 4 tratou a primeira rodada

Fontes: `data/raw_v2/sar_cantareira_diario.csv` (inspeção de leitura),
`pipeline_v2/exp4_outros_reservatorios.py`, `reports_v2/exp4/exp4_skill.csv`.

O Paiva Castro (código 29004) tem volume útil desde 1984-01-01 (14.153 dias com valor,
até 2026-10-01), mesma janela dos outros (o Jaguari-Jacareí tem 15.615 dias). Portanto NÃO
houve fase sem dado de treino: a premissa "se não havia treino" não se aplica a este
reservatório.

Mesmo assim, o mecanismo do Exp 4 já cobre esse caso: `folds_proporcionais` reserva os
primeiros 40% da série (sem NaN de volume) como treino mínimo antes do primeiro fold de
validação, e `validar()` pula (continue) qualquer fold com menos de 200 linhas de treino
ou menos de 30 de validação, contando em `n_folds` só os folds completados. O Paiva Castro
completou os 4 folds (n_folds = 4). Resultado: skill vs B3 negativo
(-0,18 / -0,08 / -0,21 em h30 / h60 / h90): o pipeline roda inalterado, mas não supera o B3
nesse reservatório pequeno (capacidade de referência 7,6 hm3).

## 1f. Skill contra o B2 (climatologia) existe em CSV?

Fontes: `reports_v2/resultados/resumo_regressao.csv` e `metricas_validacao.csv`.

Não existe coluna de skill contra o B2. Os CSVs trazem só `skill_B1` e `skill_B3`. O B2
aparece apenas como baseline, com MAE/RMSE próprios (CLIMA_ESTADO: MAE 22,942 / 22,991 / 23,072 ;
RMSE 27,646 / 27,663 / 27,706 em h30 / h60 / h90) e com skill_B1 fortemente negativo
(-4,62 / -2,02 / -1,16), ou seja, a climatologia é muito pior que a persistência. A régua do
trabalho é sempre B1 e B3. Nada foi calculado de novo.

## 1g. Sintético: skill médio por número de secas injetadas (N)

Fonte: `reports_v2/sintetico/grade_phief.csv` (origem "grade" traz N = 3, 10, 30; origem
"N0" traz N = 0), colunas `skill_30/60/90`. Média das linhas, com o mesmo conjunto de phi
(0; 0,5; 0,8; 0,95) e as mesmas sementes para todos os N (comparação equilibrada):

| N | skill h30 | skill h60 | skill h90 | média dos horizontes |
|---|---|---|---|---|
| 0 | 0,607 | 0,431 | 0,306 | 0,448 |
| 3 | 0,566 | 0,381 | 0,234 | 0,394 |
| 10 | 0,578 | 0,403 | 0,267 | 0,416 |
| 30 | 0,655 | 0,491 | 0,350 | 0,498 |

ALERTA (ponto que pode mudar a narrativa, reportado sem alterar o texto): o skill NÃO
cresce de forma monotônica com N. Em N = 3 e N = 10 o skill fica ABAIXO de N = 0; só em
N = 30 ele é claramente o maior. A frase "o ganho cresceu com o número de secas" não se
sustenta como monotônica. O que o sintético mostra com nitidez é que o skill cresce com a
persistência da chuva (phi_ef); o número de secas só ajuda de forma clara no extremo
(N = 30). Sugiro reescrever a frase no texto nessa direção. Decisão de redação é do Victor.
