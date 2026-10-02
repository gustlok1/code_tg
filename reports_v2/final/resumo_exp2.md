# Exp 2 — Validação temporal (clima vs clima+estado)

Objetivo: medir, com validação honesta (janela crescente, 4 folds, purga, baselines), se o clima
sozinho ou somado ao estado do reservatório prevê o volume. Código: `pipeline_v2/02_treino_v2.py`,
`modelagem_v2.py`. Figuras/CSVs em `reports_v2/resultados/`.

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

## PR-AUC de entrada (limiar 40) com IC95% contra a prevalência
IC95% por bootstrap em blocos de 90 dias na validação OOF (`prauc_entrada_ic.csv`). XGBRes é o modelo
residual (Exp 3) com a crise definida pelo volume previsto abaixo do limiar; B3 é a régua. A
prevalência (fração de dias de entrada que entram em crise) é o piso de uma classificação aleatória e
muda com o horizonte.

| horizonte | prevalência | B3 PR-AUC [IC95%] | XGBRes PR-AUC [IC95%] |
|---|---|---|---|
| 30 | 0,062 | 0,805 [**0,584**; 0,983] | 0,906 [**0,795**; 0,969] |
| 60 | 0,123 | 0,856 [**0,674**; 0,974] | 0,867 [**0,699**; 0,962] |
| 90 | 0,171 | 0,844 [**0,634**; 0,971] | 0,935 [**0,848**; 0,976] |

Leitura: em 30, 60 e 90 dias, o limite inferior do IC95% dos dois modelos fica muito acima da
prevalência (0,584 vs 0,062; 0,674 vs 0,123; 0,634 vs 0,171 para o B3; 0,795/0,699/0,848 para o
XGBRes). Ou seja, tanto o B3 quanto o XGBRes preveem a entrada na crise de forma significativamente
melhor que o acaso. Os intervalos de B3 e XGBRes se sobrepõem, então a diferença entre os dois não é
significativa; a utilidade de prever a entrada vem sobretudo do baseline forte (persistência sazonal).

## Interpretação
- H2 confirmada: clima SOZINHO não prevê o nível do volume (skill fortemente negativo); o estado do
  reservatório é indispensável.
- Com estado, os modelos EMPATAM com o B3 (não o superam na validação): o B3 é uma régua forte.
- O recorte "entrada" (prever a entrada na crise) é o honesto; aí os modelos com estado ficam juntos
  do B3 e a regra SPI-6 é fraca.
