# Exp 4 — O mesmo pipeline em outros reservatórios (H4)

Objetivo: mostrar que o arcabouço roda inalterado em reservatórios distintos. Código:
`pipeline_v2/exp4_outros_reservatorios.py`. Saídas em `reports/v2/exp4/`.

## O que foi feito
O pipeline (B3 + XGBRes residual, folds + purga, SPI no treino) rodou nos 4 reservatórios INDIVIDUAIS
do Cantareira, que são reservatórios distintos com dinâmicas bem diferentes (Jaguari-Jacareí 808 hm³ a
Paiva Castro 8 hm³), cada um com sua série de volume/vazões e seu clima. Cobertura 1984-2026.

## Resultado — skill vs B3 na validação (`exp4_skill.png`, `exp4_skill.csv`)
| reservatório | capac. (hm³) | h30 | h60 | h90 | vol mín (%) |
|---|---|---|---|---|---|
| Jaguari-Jacareí | 808 | +0,113 | +0,049 | +0,005 | −19,2 |
| Cachoeira | 70 | −0,032 | +0,043 | +0,006 | −3,1 |
| Atibainha | 96 | −0,409 | −0,431 | −0,300 | −106,8 |
| Paiva Castro | 8 | −0,181 | −0,080 | −0,210 | 8,7 |

## Interpretação
H4 parcial: o pipeline é PORTÁVEL (roda inalterado em 4 reservatórios), mas a utilidade depende da
dinâmica — empata/supera o B3 nos maiores e mais estáveis (Jaguari, Cachoeira) e fica abaixo nos
pequenos e voláteis (Atibainha, Paiva Castro). Nota de dado: o "volume útil %" dos pequenos vai a
valores extremos (Atibainha mínimo −106,8%), o que torna a série ruidosa; vale entender/tratar esse
percentual antes de usar esses reservatórios.

## Limitação (time-box, 1 dia)
Reservatórios FORA do Cantareira (Nordeste e SIN do SAR) não entraram porque os endpoints
`/sar0/Medicao` e `/sar0/MedicaoSin` servem a série por AJAX com estado de sessão (o HTML estático traz
só o cabeçalho), ao contrário do `/sar0/MedicaoCantareira`. O harness já aceita qualquer reservatório;
falta a coleta. Caminhos para depois: endpoint AJAX interno, Claude-in-Chrome dirigindo a página, ou
dados abertos da ONS.
