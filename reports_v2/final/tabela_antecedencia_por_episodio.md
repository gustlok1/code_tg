# Antecedência do alerta por episódio (limiar 40%)

Dias de antecedência do primeiro alerta (volume previsto < limiar) antes do início de cada episódio, por modelo e horizonte. Diferença = XGBRes − B3 (positivo: o XGBRes antecipa mais). Janela de busca de 120 dias; onde o alerta já está ativo no começo da janela, a antecedência satura em 120 e é marcada como "alerta contínuo".

Fontes: validação em `reports_v2/resultados/antecedencia_pareada_por_episodio.csv`; teste em `reports_v2/final/antecedencia_teste.csv`. Nenhum número foi recalculado.

### Validação (1984–2022, janela crescente em 4 folds) — 10 episódios

| Início do episódio | Horizonte (d) | XGBRes (d) | B3 (d) | Diferença (d) | Observação |
|---|---|---|---|---|---|
| 2003-08-15 | 30 | 25 | 20 | +5 | entra no Wilcoxon |
| 2003-08-15 | 60 | 49 | 44 | +5 | entra no Wilcoxon |
| 2003-08-15 | 90 | 69 | 68 | +1 | entra no Wilcoxon |
| 2005-10-20 | 30 | 18 | 26 | -8 | entra no Wilcoxon |
| 2005-10-20 | 60 | 82 | 62 | +20 | entra no Wilcoxon |
| 2005-10-20 | 90 | 113 | 106 | +7 | entra no Wilcoxon |
| 2006-10-30 | 30 | 23 | 22 | +1 | entra no Wilcoxon |
| 2006-10-30 | 60 | 62 | 48 | +14 | entra no Wilcoxon |
| 2006-10-30 | 90 | 90 | 80 | +10 | entra no Wilcoxon |
| 2007-08-29 | 30 | 65 | 60 | +5 | entra no Wilcoxon |
| 2007-08-29 | 60 | 111 | 93 | +18 | entra no Wilcoxon |
| 2007-08-29 | 90 | 120 | 118 | +2 | alerta contínuo (teto 120 d); entra no Wilcoxon |
| 2013-10-01 | 30 | 17 | 18 | -1 | entra no Wilcoxon |
| 2013-10-01 | 60 | 48 | 40 | +8 | entra no Wilcoxon |
| 2013-10-01 | 90 | 61 | 78 | -17 | entra no Wilcoxon |
| 2018-07-28 | 30 | 36 | 22 | +14 | entra no Wilcoxon |
| 2018-07-28 | 60 | 88 | 52 | +36 | entra no Wilcoxon |
| 2018-07-28 | 90 | 99 | 78 | +21 | entra no Wilcoxon |
| 2019-11-06 | 30 | 24 | 16 | +8 | entra no Wilcoxon |
| 2019-11-06 | 60 | 96 | 49 | +47 | entra no Wilcoxon |
| 2019-11-06 | 90 | 103 | 109 | -6 | entra no Wilcoxon |
| 2020-10-06 | 30 | 32 | 19 | +13 | entra no Wilcoxon |
| 2020-10-06 | 60 | 78 | 47 | +31 | entra no Wilcoxon |
| 2020-10-06 | 90 | 97 | 85 | +12 | entra no Wilcoxon |
| 2021-08-10 | 30 | 39 | 28 | +11 | entra no Wilcoxon |
| 2021-08-10 | 60 | 82 | 58 | +24 | entra no Wilcoxon |
| 2021-08-10 | 90 | 108 | 90 | +18 | entra no Wilcoxon |
| 2022-06-26 | 30 | 43 | 29 | +14 | entra no Wilcoxon |
| 2022-06-26 | 60 | 66 | 50 | +16 | entra no Wilcoxon |
| 2022-06-26 | 90 | 86 | 79 | +7 | entra no Wilcoxon |

O teste de Wilcoxon pareado da antecedência (XGBRes vs B3, limiar 40) usa estes 10 episódios em cada horizonte. Resultado (de `antecedencia_pareada.csv`): mediana da diferença +6,5 d em h30 (p = 0,039), +19 d em h60 (p = 0,002) e +7 d em h90 (p = 0,10).

### Teste (2023-01-01 em diante, modelo congelado) — 2 episódios

| Início do episódio | Horizonte (d) | XGBRes (d) | B3 (d) | Diferença (d) | Observação |
|---|---|---|---|---|---|
| 2025-08-06 | 30 | 25 | 19 | +6 |  |
| 2025-08-06 | 60 | 48 | 40 | +8 |  |
| 2025-08-06 | 90 | 75 | 64 | +11 |  |
| 2026-06-02 | 30 | 120 | 120 | +0 | alerta contínuo (teto 120 d) |
| 2026-06-02 | 60 | 120 | 120 | +0 | alerta contínuo (teto 120 d) |
| 2026-06-02 | 90 | 120 | 120 | +0 | alerta contínuo (teto 120 d) |

No teste, o limiar 40 tem apenas 2 episódios iniciando no período, e o de 2026-06-02 é "alerta contínuo" nos três horizontes (o volume já está baixo quando a janela abre). Com um único episódio informativo (2025-08-06), não há teste de Wilcoxon no conjunto de teste; ali o XGBRes antecipa o alerta em +6, +8 e +11 dias sobre o B3 em h30, h60 e h90.
