# Capitulo 4 — Analise e Resultados (rascunho)

Rascunho para integracao ao texto. Numeros vem dos CSVs em reports_v2/. Figuras em
reports_v2/final/figuras/ (ver indice_figuras.md). A regua de comparacao e o baseline B3
(persistencia mais variacao sazonal). O teste (2023 em diante) foi aberto uma unica vez,
apos o congelamento das escolhas; nada mudou depois.

## 4.1 Diagnostico da versao 1 (Exp 0)

O que foi feito. Reproduzimos os defeitos da v1 a partir dos seus proprios dados, sem alterar a v1.
Quatro figuras resumem o diagnostico (Figuras 1 a 4).

Numeros. A chuva agregada da v1 cresce com o numero de estacoes empilhadas: 3.438 mm em 2003 (cerca de
3,9 estacoes) e 50.476 mm em 2019 (cerca de 41,5 estacoes). A radiacao agregada (RAD_SUM) fica igual a
zero em todo o periodo de teste (2020 a 2024) por troca de grafia da coluna, e inflada no treino. Na
validacao cruzada, os positivos por fold sao [0, 0, 0, 15, 0]: quatro de cinco folds sem nenhum positivo.
Uma regra de limiar sobre o SPI30 tem AUC-ROC 0,785, 0,755 e 0,752 em y30, y60 e y90, acima de todos os
modelos da v1 (o melhor, XGBoost, fica em 0,700, 0,418 e 0,592).

Interpretacao. A v1 nao media o clima, media um artefato de agregacao. A avaliacao estava quebrada
(folds sem positivo zeram o F1 do CV) e o modelo nao superava uma regra simples. Esse diagnostico
motivou a reconstrucao: alvo objetivo (volume do SAR), sem agregacao indevida, validacao com purga e
baselines explicitos.

## 4.2 Compreensao dos dados (EDA)

O que foi feito. Montamos um dataset diario continuo de 1984-01-01 a 2026-10-02 (15.616 dias, 72
colunas, 0,4% de celulas ausentes), com volume do sistema (SAR/ANA), clima da bacia por duas fontes
(ERA5-Land e NASA POWER) e episodios de crise detectados automaticamente. Figuras 5 e 6.

Numeros. O volume util do sistema chega a -23,21% em 2015-02-02 (uso da reserva tecnica), valor real
que foi mantido. A deteccao automatica encontra cinco episodios abaixo de 30% (1986, 2003-04, 2013-16,
2021-22 e 2025-26) e dezessete abaixo de 40%. A chuva anual das duas fontes concorda na maior parte da
serie, com uma excecao: em 1999 o NASA POWER tem um pico anomalo (cerca de 3.380 mm contra cerca de
1.600 mm do ERA5). Por isso o ERA5 foi adotado como fonte principal de chuva; o POWER entrou so em um
teste de robustez.

Interpretacao. Os dados reconstruidos tem o sinal que a v1 nao tinha: o volume do sistema com as faixas
operacionais e episodios bem definidos. A anomalia de 1999 do POWER mostra a importancia de comparar
fontes antes de usar.

## 4.3 Experimento sintetico (Exp 1)

O que foi feito. Um gerador fisico produz 40 anos diarios de chuva, afluencia (reservatorio linear) e
volume (balanco com as faixas ANA/DAEE), com uma persistencia da chuva (phi) e um numero de secas (N)
controlados. Rodamos o mesmo pipeline nesses cenarios. Figuras 7 e 8.

Numeros. A chuva real do Cantareira tem persistencia mensal quase nula: phi real igual a 0,0129. Os
testes de sanidade confirmam o pipeline: com o alvo embaralhado o skill contra o B3 vai a zero (-0,011)
e a PR-AUC de entrada cai de 0,88 para 0,77, mas fica acima da prevalencia (0,16) porque a previsao
ainda e o B3 mais ruido e o proprio B3 ja antecipa a entrada em crise; com a chuva futura (oraculo) o
skill sobe para 0,62. O ganho do modelo sobre o B3 cresce com a persistencia efetiva da chuva (phi_ef)
e com N. O valor do embaralhamento e reprodutivel bit a bit (semente fixa por indice do fold).

Limitacao do gemeo. Um gemeo calibrado reproduz as propriedades do real: tau do reservatorio 24 dias,
ruido multiplicativo na afluencia 0,63, phi real, e chuva mensal dentro de 10% do real (media 121,6
contra 124,37 mm; desvio 84,69 contra 87,84). Mesmo assim o gemeo ainda supera o B3 em 30 e 60 dias
(skill 0,268 e 0,140) e empata em 90 dias (0,064). O real empata em todos os horizontes.

Interpretacao, em tres frentes. Estatistica: o oraculo prova que o teto de desempenho e a informacao
sobre a chuva futura, nao o algoritmo. Hidrologica: o gemeo erra para mais porque o volume sintetico e
funcao quase deterministica da afluencia; o volume real tambem depende de operacao, transferencias
entre bacias e varios reservatorios, que o gerador nao modela. Metodologica: o sintetico serve para
provar que o pipeline recupera sinal quando ele existe, nao para medir a dificuldade real da previsao.

## 4.4 Clima contra clima e estado (Exp 2)

O que foi feito. Validacao com janela crescente (4 folds) e purga, comparando dois conjuntos de
features (CLIMA e CLIMA_ESTADO) contra os baselines. Figura 9.

Numeros. Clima sozinho nao preve o nivel do volume: o XGBReg em CLIMA tem skill contra a persistencia
de -4,33, -1,90 e -1,25 em 30, 60 e 90 dias. Com o estado do reservatorio (CLIMA_ESTADO), o XGBReg
sobe para +0,28, +0,35 e +0,36, mas isso apenas empata com o B3 (skill contra o B3 de -0,036, -0,008
e -0,032). Na classificacao de entrada na crise (limiar 40), a PR-AUC do XGBClf em CLIMA_ESTADO e
0,744, 0,743 e 0,841, no mesmo nivel do B3 (0,748, 0,871 e 0,806); clima sozinho fica abaixo de 0,54 e
a regra SPI-6 menor que -1 nao passa de 0,31.

Interpretacao, em tres frentes. Estatistica: o skill do CLIMA_ESTADO contra o B3 e nulo (proximo de
zero e sem significancia). Hidrologica: o clima agregado nao ancora o nivel absoluto do reservatorio;
quem carrega o nivel e o proprio estado do reservatorio, que o B3 ja usa. Metodologica: a confirmacao
da hipotese H2 (o estado agrega ganho mensuravel sobre o clima) vem do contraste CLIMA contra
CLIMA_ESTADO, nao de bater o B3.

## 4.5 Modelos contra a regua B3 (Exp 3)

O que foi feito. Um modelo residual preve r_h = vol(t+h) - B3(t); a previsao final e B3 mais o residual.
Avaliamos geral, em subconjuntos (seca, por SPI-12 menor que -1; e pre-episodio, de 180 dias antes a 30
depois de cada episodio) com IC95% por bootstrap em blocos de 90 dias, mais o LSTM e um teste pareado
da antecedencia. Figuras 10 e 11.

Numeros. O skill do XGBRes contra o B3 e 0,004, 0,022 e 0,037 em geral (todos os IC95% cruzam zero).
Na seca e -0,139, -0,015 e -0,080; no pre-episodio e 0,044, 0,096 e -0,104. Em nenhum subconjunto ou
horizonte o intervalo exclui o zero. O LSTM empata com o XGBRes (skill medio 0,022 contra 0,021; 409 s
de execucao). O unico ganho significativo e de antecedencia: no limiar 40, o XGBRes antecipa o alerta
em +6,5 dias a 30 dias (Wilcoxon p = 0,039) e +19 dias a 60 dias (p = 0,002); a 90 dias a diferenca
nao e significativa (+7 dias, p = 0,10). O custo e 3 a 4 episodios de falso-alarme em 20 anos, contra
zero do B3.

Interpretacao, em tres frentes. Estatistica: o modelo nao supera o B3 no erro de forma confiavel;
superar aparece so na antecedencia, com significancia em 30 e 60 dias. Hidrologica: a seca e o
pre-episodio sao justamente onde o clima recente poderia ajudar, e e ali que o B3 ja captura quase
tudo; o ganho vira antecedencia porque o modelo reage antes a sinais de deficit. Metodologica: a regua
B3 (persistencia mais sazonalidade) e forte e deve ser sempre reportada; sem ela, o F1 e a AUC isolados
enganariam.

## 4.6 Resultado no teste e diagnosticos pos-hoc

O que foi feito. Congelamos conjunto, modelo, hiperparametros e thresholds na validacao, commitamos o
congelamento, e so entao abrimos o teste (2023 em diante) uma unica vez. Depois, calculamos
diagnosticos pos-hoc que nao alteram o modelo. Figuras 12 e 13.

Numeros. No teste, o XGBRes supera o B3: skill contra o B3 de 0,240, 0,229 e 0,167 em 30, 60 e 90 dias
(MAE 2,24, 3,75 e 5,22 contra 2,95, 4,87 e 6,27 do B3). O IC95% por bootstrap em blocos de 90 dias e
[0,132; 0,372] em 30 dias e [0,079; 0,420] em 60 dias, ambos acima de zero; em 90 dias e [-0,052; 0,378]
e cruza zero. A vantagem aparece nos dois recortes de periodo: 0,242 em 2023-2024 e 0,237 em 2025-2026
(30 dias). Um B3 ajustado so em 2017-2022 (pos-Resolucao 925) explica apenas 12% a 25% da vantagem; o
resto e ganho real. Na crise de 2025-26 (episodio iniciando em 2025-08-06, limiar 40), o XGBRes
antecipa o alerta em 25, 48 e 75 dias contra 19, 40 e 64 do B3, quase sem falso-alarme.

Interpretacao. Na validacao, que e a evidencia principal (20 anos, 4 folds, 10 episodios), o modelo
empata com o B3. No teste, que e uma realizacao unica (2023-2026, com a depleção de 2025-26), o modelo
supera o B3 de forma significativa em 30 e 60 dias e antecipa a crise recente. Os dois resultados sao
coerentes e honestos: a validacao mede o tipico; o teste confirma a direcao em um periodo favoravel.
Nada foi re-selecionado apos a abertura do teste.

## 4.7 Generalizacao (Exp 4)

O que foi feito. Rodamos o mesmo pipeline, sem alterar nada, nos quatro reservatorios individuais do
Sistema Cantareira (Jaguari-Jacarei 808 hm3, Cachoeira 70, Atibainha 96, Paiva Castro 8), que sao
reservatorios distintos com series e climas proprios. Figura 14.

Numeros. Skill contra o B3 na validacao: Jaguari-Jacarei +0,113, +0,049 e +0,005; Cachoeira -0,032,
+0,043 e +0,006; Atibainha -0,409, -0,431 e -0,300; Paiva Castro -0,181, -0,080 e -0,210.

Interpretacao, em tres frentes. Estatistica: o skill e positivo ou nulo nos dois maiores e negativo nos
dois menores. Hidrologica: os reservatorios pequenos tem dinamica mais rapida e o percentual de volume
util vai a valores extremos (Atibainha chega a -106,8%), o que torna a serie ruidosa e dificil de
prever. Metodologica: o pipeline e portavel (roda inalterado), mas a utilidade depende da dinamica do
reservatorio e da qualidade da serie; reservatorios fora do Cantareira ficam como trabalho futuro
porque os endpoints do SAR servem a serie por AJAX com sessao.

## 4.8 Aplicacao web

O que foi feito. Um app Streamlit (app_v2) usa o modelo congelado para mostrar o volume atual, a
previsao em 30, 60 e 90 dias com faixa de incerteza e a faixa operacional ANA/DAEE prevista, alem de
paginas de backtest das crises conhecidas e do experimento sintetico (com selo de dado sintetico) e
uma atualizacao por API (SAR, ERA5 e NASA POWER).

Numeros. Na ultima observacao disponivel (2026-10-01), o volume e 40,9% (faixa Atencao) e a previsao
e 40,8%, 44,4% e 48,7% em 30, 60 e 90 dias, com a faixa de incerteza vinda dos residuos do modelo.

Interpretacao. O app entrega o resultado de forma util e honesta: mostra a incerteza, separa o dado
sintetico do real e nao repete a afirmacao incorreta da v1 sobre a estacao A701.
