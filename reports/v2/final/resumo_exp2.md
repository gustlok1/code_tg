# Exp 2 — Validação temporal (clima vs clima+estado)

Objetivo: medir, com validação honesta (janela crescente, 4 folds, purga, baselines), se o clima
sozinho ou somado ao estado do reservatório prevê o volume. Código: `pipeline_v2/02_treino_v2.py`,
`modelagem_v2.py`. Figuras/CSVs em `reports/v2/resultados/`.

## Regressão do volume — skill contra a persistência (B1), média dos folds
| conjunto / modelo | h30 | h60 | h90 |
|---|---|---|---|
| B3 (persistência + sazonal) | +0,31 | +0,36 | +0,38 |
| CLIMA_ESTADO / Ridge | +0,32 | +0,32 | +0,31 |
| CLIMA_ESTADO / XGBReg | +0,28 | +0,35 | +0,36 |
| CLIMA (só clima) / Ridge | −3,49 | −1,42 | −0,74 |
| CLIMA (só clima) / XGBReg | −4,33 | −1,90 | −1,25 |

Figura central: `skill_por_horizonte.png`.

## Classificação (limiar 40, recorte ENTRADA) — PR-AUC médio
| conjunto / modelo | h30 | h60 | h90 |
|---|---|---|---|
| B3 | 0,748 | 0,871 | 0,806 |
| CLIMA_ESTADO / XGBClf | 0,744 | 0,743 | 0,841 |
| CLIMA (só clima) | ≤ 0,30 | ≤ 0,45 | ≤ 0,54 |
| regra SPI-6 < −1 | 0,116 | 0,216 | 0,311 |

## Interpretação
- H2 confirmada: clima SOZINHO não prevê o nível do volume (skill fortemente negativo); o estado do
  reservatório é indispensável.
- Com estado, os modelos EMPATAM com o B3 (não o superam na validação): o B3 é uma régua forte.
- O recorte "entrada" (prever a entrada na crise) é o honesto; aí os modelos com estado ficam juntos
  do B3 e a regra SPI-6 é fraca.
