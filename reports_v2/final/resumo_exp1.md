# Exp 1 — Experimento sintético (DADOS SINTÉTICOS)

Objetivo: em ambiente controlado, mostrar (a) que o pipeline recupera um sinal conhecido, (b) que o ML
só supera o B3 quando a chuva futura tem componente previsível do passado, e (c) como o desempenho
muda com o número de crises N. Código: `pipeline_v2/sintetico_v2.py`, `04_experimento_sintetico.py`,
`04b_sintetico_ajuste.py`. Figuras em `reports_v2/sintetico/` (todas com marca "DADOS SINTÉTICOS").

## Calibração com o real
φ real (persistência efetiva da chuva mensal) = 0,0129; σ real = 0,6944. O Cantareira real está em
φ ≈ 0: a chuva não tem persistência mensal.

## Resultados
- Sanidade (`sanidade_sintetico.csv`, `barras_sanidade.png`): com o alvo embaralhado o skill vs B3 ≈ 0
  (−0,0114); a PR-AUC de entrada cai de 0,88 (modelo normal) para 0,77, mas fica acima da prevalência
  (0,16) porque a previsão ainda é B3 mais ruído e o próprio B3 já antecipa a entrada em crise; o
  oráculo (chuva futura t..t+h) dá skill +0,62. Conclusão: o teto é a informação sobre a chuva futura,
  não o algoritmo. O valor do embaralhamento agora é reprodutível bit a bit (semente fixa derivada do
  índice do fold, `sintetico.shuffle_seed_base` no config; antes vinha de `hash()` randomizado por
  processo).
- Persistência efetiva (`skill_vs_phief.png`): o skill vs B3 cresce com φ_ef e com N. φ_param=0 com N=0
  dá φ_ef = 0,015 (≈ real); as secas injetadas inflam φ_ef (N=30 → 0,37). O ganho salta em φ alto
  (φ=0,95 → skill ~0,43 em h=90).
- Gêmeo calibrado (φ e τ reais, N=0, 10 seeds): τ = 24 d, ruído de afluência σ = 0,63, σ da anomalia
  calibrado = 0,25. Chuva mensal dentro de 10% do real (média −2,2%, desvio −3,6%). Skill vs B3:
  h30 +0,268 [+0,211; +0,311], h60 +0,140, h90 +0,064.

## Limitação (decisão do Victor, aceita)
Mesmo calibrado, o gêmeo AINDA supera o B3 em h30 e h60 e empata em h90 — o real empata em tudo. Causa:
o volume sintético é previsível demais a partir da chuva recente porque o gerador NÃO modela a operação
do sistema (política de retirada variável, transferências entre bacias, dinâmica de vários reservatórios).
Consequência para o texto: o Exp 1 prova que o pipeline recupera sinal quando ele existe, mas NÃO é um
substituto do real para medir a dificuldade da previsão.
