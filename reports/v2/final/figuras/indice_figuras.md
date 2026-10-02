# Indice de figuras (capitulo 4)

| Nº | Arquivo | Titulo | Leitura |
|---|---|---|---|
| 1 | `fig_01_v1_chuva_vs_estacoes.png` | Chuva agregada da v1 contra numero de estacoes | A chuva somada vai de 3.438 mm (3,9 estacoes, 2003) a 50.476 mm (41,5 estacoes, 2019): media um artefato de agregacao, nao o clima. |
| 2 | `fig_02_v1_rad_zerada.png` | Radiacao zerada no teste da v1 | Por troca de grafia da coluna, RAD_SUM vale zero em 2020-2024 (teste) e fica inflada no treino. |
| 3 | `fig_03_v1_folds_sem_positivo.png` | Folds sem positivo na v1 | Os positivos por fold sao [0,0,0,15,0]: o threshold degenera e o F1 do CV e zero. |
| 4 | `fig_04_v1_spi_vs_modelos.png` | Regra do SPI contra os modelos da v1 | A regra -SPI30 tem AUC 0,785/0,755/0,752, acima dos modelos (melhor XGBoost 0,700). |
| 5 | `fig_05_eda_volume_faixas.png` | Volume do sistema com faixas e episodios | O sistema chega a -23,2% em 2015 (reserva tecnica). Cinco episodios < 30%: 1986, 2003-04, 2013-16, 2021-22 e 2025-26. |
| 6 | `fig_06_eda_chuva_era5_vs_power.png` | Chuva anual ERA5 contra POWER | As fontes concordam, menos em 1999, quando o POWER tem um pico anomalo (~3.380 mm vs ~1.600 do ERA5). |
| 7 | `fig_07_sintetico_skill_phief.png` | Skill sintetico por persistencia efetiva | DADOS SINTETICOS. O Cantareira real esta em phi ~ 0; o gemeo calibrado (estrela) ainda supera o B3 em h90, limitacao discutida no texto. |
| 8 | `fig_08_sintetico_sanidade.png` | Sanidade do sintetico | DADOS SINTETICOS. Embaralhado skill -0,00; oraculo +0,62: o teto e a informacao sobre a chuva futura, nao o algoritmo. |
| 9 | `fig_09_exp2_skill_horizonte.png` | Clima contra clima e estado (Exp 2) | CLIMA sozinho tem skill negativo; CLIMA_ESTADO fica junto do B3 (skill vs B1 ~0,3), sem supera-lo. |
| 10 | `fig_10_exp3_skill_condicional.png` | Skill condicional do residual (Exp 3) | Em geral, seca e pre-episodio, todos os IC95% do XGBRes cruzam zero: nao supera o B3 onde importa. |
| 11 | `fig_11_exp3_antecedencia_pareada.png` | Antecedencia pareada (Exp 3) | Mediana da diferenca: +6,5 d (h30, p=0,039), +19 d (h60, p=0,002), +7 d (h90, p=0,10). Custo: 3-4 episodios de falso-alarme em 20 anos contra zero do B3. |
| 12 | `fig_12_teste_skill_ic.png` | Resultado do teste com IC95% e por periodo | Skill +0,240 [0,132;0,372] (h30) e +0,229 [0,079;0,420] (h60) significativos; h90 cruza zero. A vantagem aparece em 2023-2024 e 2025-2026. |
| 13 | `fig_13_teste_backtest_2025_26.png` | Previsao no teste da crise de 2025-26 | No teste, a 90 dias, o XGBRes acompanha a queda melhor que o B3; antecipa o alerta em ~11 dias. |
| 14 | `fig_14_exp4_reservatorios.png` | Generalizacao para outros reservatorios (Exp 4) | O pipeline roda inalterado. Skill vs B3: Jaguari +0,11 (h30), Cachoeira ~0, Atibainha -0,41, Paiva Castro -0,18. |
