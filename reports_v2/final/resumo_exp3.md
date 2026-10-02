# Exp 3 — Modelo residual sobre o B3, LSTM e o teste

Objetivo: tentar superar o B3 com um modelo residual (prevê r_h = vol(t+h) − B3(t)), avaliar onde
importa (seca e pré-episódio) e, por fim, abrir o teste. Código: `pipeline_v2/02b_analise_exp3.py`,
`analise_exp3.py`, `02c_lstm.py`, `lstm_v2.py`, `02d_antecedencia_pareada.py`, `03_congelar_e_testar.py`,
`03b_teste_diagnostico.py`. Figuras/CSVs em `reports_v2/resultados/` e `reports_v2/final/`.

## Validação (régua = skill vs B3)
- Residual XGBRes vs B3 (skill_B3 médio): h30 −0,004; h60 +0,018; h90 +0,016. RidgeRes pior.
- Skill condicional (`skill_condicional.png`): em geral, seca (SPI-12 < −1) e pré-episódio, TODOS os
  IC95% do XGBRes cruzam zero. Não supera o B3 de forma confiável onde importa.
- LSTM (`skill_lstm.png`): empate técnico com o XGBRes (skill_B3 médio 0,022 vs 0,021; 409 s). IC95%
  cruzam zero em tudo.
- Antecedência pareada XGBRes − B3 (limiar 40): +6,5 d em h30 (Wilcoxon p=0,039), +19 d em h60
  (p=0,002); custo: 3-4 episódios de falso-alarme em 20 anos (B3 tem 0).

## Teste (2023+), modelos CONGELADOS — aberto uma única vez
- Regressão: o XGBRes SUPERA o B3 no teste, skill_B3 +0,240 (h30), +0,229 (h60), +0,167 (h90).
- Diagnósticos pós-hoc (`teste_diagnostico.csv`, não alteram o modelo):
  - IC95% (bootstrap blocos 90 d): h30 [0,132; 0,372] e h60 [0,079; 0,420] SIGNIFICATIVOS; h90
    [−0,052; 0,378] não.
  - Por período: a vantagem aparece em 2023-2024 e em 2025-2026 (não é um único evento).
  - B3 ajustado só em 2017-2022 explica apenas 12-25% da vantagem; o resto é ganho real.
- Crise de 2025-26 (limiar 40): XGBRes antecipa +6 d (h30), +8 d (h60), +11 d (h90) sobre o B3.

## Interpretação (enquadramento aprovado)
A evidência principal é a validação (20 anos, 4 folds, 10 episódios): o modelo EMPATA com o B3, uma
régua forte, e só compra antecedência pagando falso-alarme. No teste (realização única, 2023-2026) o
modelo SUPEROU o B3 de forma significativa em 30 e 60 dias e antecipou a crise recente. Resultado
honesto: apoio parcial a H1/H2; o teste confirma a direção, não a prova sozinho. Nada muda após aberto.
