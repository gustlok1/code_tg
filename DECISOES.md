# Decisões de projeto (TG2 v2)

Registro das decisões técnicas relevantes, conforme a seção 5 do CLAUDE.md.
Formato: data, decisão, motivo, alternativa descartada.

## 2026-10-02 — Arquitetura modular e teste intocável
- Decisão: lógica em módulos importáveis (`dataset_v2.py`, `modelagem_v2.py`, `analise_exp3.py`,
  `lstm_v2.py`, `sintetico_v2.py`, `transformadores.py`) e orquestradores numerados
  (`01`, `02`, `02b`, `02c`, `02d`, `04`, `04b`). O teste (2023-01-01 em diante) só abre no `03`.
- Motivo: testabilidade (pytest importa os módulos; um `0X_*.py` não é importável por nome) e
  rigor científico (congelar antes de abrir o teste).
- Alternativa descartada: colocar tudo nos orquestradores; dificultaria o pytest e o congelamento.

## 2026-10-02 — Fontes de dados (coleta real)
- Decisão: SAR/ANA (volume do sistema, desde 1984), ERA5-Land via Open-Meteo e NASA POWER,
  média de bacia sobre os 4 reservatórios (Jaguari-Jacareí, Cachoeira, Atibainha, Paiva Castro).
- Motivo: corrigir o defeito da v1 (estações do INMET empilhadas sem id, chuva somada). O SAR dá
  o alvo objetivo (volume útil) e a ANA/DAEE 925 define as faixas de crise.
- Alternativa descartada: continuar com o INMET agregado da v1.
- Nota: volume negativo (reserva técnica, 2014-2015, mínimo -23,21% em 2015-02-02) é mantido como
  dado real, não cortado.

## 2026-10-02 — Alvo e baselines
- Decisão: alvo = volume útil do sistema em t+30/60/90; crise principal < 40%, grave < 40/30%.
  Baselines sempre reportados: B1 (persistência), B2 (climatologia), B3 (persistência + variação
  sazonal). O skill contra o B3 é a régua.
- Motivo: substituir o rótulo manual da v1 (que punia antecipação). B3 é a régua honesta e forte.
- Alternativa descartada: rótulo binário manual por datas (v1).

## 2026-10-02 — SPI/SPEI/anomalias só no treino
- Decisão: SPI, SPEI (gama) e anomalias sazonais vivem em `transformadores.py` com fit no treino
  de cada fold e transform no resto; nunca estatística de série inteira fora dali. Purga de h dias.
- Motivo: evitar vazamento (defeito clássico). Validação com janela crescente, 4 folds.
- Alternativa descartada: SPI calculado sobre a série inteira (como na v1).

## 2026-10-02 — Exp 3: modelo residual sobre o B3
- Decisão: o ML prevê o resíduo r_h = vol(t+h) - B3(t); previsão final = B3 + r_h. Avaliação
  condicional (seca por SPI-12 < -1; pré-episódio -180..+30 d) com IC95% por bootstrap em blocos
  de 90 dias.
- Motivo: comparação direta e honesta com a régua B3, onde importa (seca e pré-crise).
- Resultado: XGBRes e LSTM empatam com o B3 (todos os IC95% cruzam zero). Reportado como é.

## 2026-10-02 — LSTM com orçamento
- Decisão: LSTM (1 camada, 32 unidades, janelas de 180 d, 3 seeds, early stopping) no mesmo
  arcabouço, com orçamento de tempo. Se não superar o XGBRes na validação, registrar e seguir.
- Resultado: empate técnico (skill_B3 médio 0,0219 vs 0,0211; 409 s). Não superou de forma confiável.

## 2026-10-02 — Exp 1 sintético: persistência efetiva e gêmeo calibrado
- Problema: a grade original mostrava skill > 0 do ML sobre o B3 mesmo em phi=0, contradizendo o
  real. Causa: as secas injetadas (regimes de 6 a 12 meses) criam persistência mesmo com phi=0, então
  o phi do gerador não é a persistência efetiva da série.
- Decisão 1: medir a persistência EFETIVA phi_ef = autocorrelação lag-1 das anomalias mensais da
  chuva, com a MESMA função usada no real (`phi_lag1_mensal`), e refazer as figuras com phi_ef no eixo.
  A grade antiga foi mantida (continua útil para o efeito de N).
- Decisão 2: acrescentar cenários N=0 (crises só da variabilidade natural) para phi em {0; 0,5; 0,8; 0,95}.
- Decisão 3: gêmeo calibrado — tau do reservatório linear ajustado por mínimos quadrados à afluência
  real do SAR (1984-2022); ruído multiplicativo log-normal na afluência com o desvio do resíduo desse
  ajuste; phi real; N=0; 10 seeds.
- Decisão 4: o sigma da anomalia do gêmeo é CALIBRADO numericamente ao desvio mensal real
  (`calibrar_sigma_anom`), não usado diretamente o sigma_real do AR(1) log. Motivo: o sigma_real do
  AR(1) em log superdispersa a chuva mensal (desvio ~+50%) por causa do ruído diário
  (Bernoulli x gama); o gêmeo precisa reproduzir média E desvio mensal reais dentro de 10%.
- Decisão 5: o sigma do ruído multiplicativo da afluência é calculado só nas linhas com afluência e
  previsão positivas (a afluência do SAR tem 9 valores negativos espúrios, min -29,96 m3/s, que
  quebravam o log).
- Resultado do gêmeo (a reportar com os números finais do 04b): mesmo calibrado, o ML ainda supera o
  B3 em horizontes curtos (h=30), ao contrário do real que empata. Propriedade do gerador que explica:
  o volume sintético é função quase determinística da afluência (mesmo com ruído), enquanto o volume
  real é governado também por operação (retiradas por política, transferências, múltiplos reservatórios)
  não dirigida pela chuva recente. Reportado como é (rigor científico, seção 7).
- LIMITAÇÃO ACEITA (decisão do Victor, 2026-10-02): o resultado do gêmeo é aceito como está
  (supera o B3 em h30 e h60, empata em h90). Fica registrado como LIMITAÇÃO do gerador: o volume
  sintético é previsível demais a partir da chuva recente porque não modela a operação do sistema
  (política de retirada variável, transferências entre bacias, dinâmica de múltiplos reservatórios).
  Consequência para o texto: o sintético prova que o pipeline recupera sinal quando ele existe
  (objetivo do Exp 1), mas NÃO é um substituto do real para medir a dificuldade da previsão. Também
  deve constar no resumo do Exp 1 (reports/v2/final/resumo_exp1.md). Não investir mais tempo no gerador.

## 2026-10-02 — Versionamento
- Decisão: branch `tg2-v2`; `.gitignore` exclui `data/raw_v2/`, `data/processed_v2/`, `data/sintetico/`,
  `_auditoria_tmp/`, `.venv/`, `.env`. `CLAUDE.md` copiado para dentro do repo (`code_tg/CLAUDE.md`)
  para ficar versionado com a v2; o original em `C:\Dev\TG\CLAUDE.md` permanece.
- Motivo: a v2 nunca tinha sido commitada; o primeiro commit em `tg2-v2` consolida o pipeline_v2 já
  construído (Exp 1 a 3 + LSTM) mais o sintético concluído.
- Nota: `data/`, `models/` e `reports/v1_*`/`v2_threshold_otimizado/` pesados já estão no HEAD (commits
  do Gustavo). Não são tocados; a remoção do `.venv` do índice fica para o item 8 (higiene).

## 2026-10-02 — Congelamento e abertura do teste (03)
- Decisão: duas fases. Fase 1 grava `reports/v2/final/congelamento.json` (conjunto CLIMA_ESTADO +
  SPI/SPEI + anomalias; XGBRes por horizonte; thresholds de alerta B3/XGBRes nos limiares 40 e 30,
  escolhidos por F1 na validação) e é commitada ANTES de abrir o teste. Fase 2 (`--abrir-teste`) abre
  o teste 2023+ uma única vez, registra em `teste_aberto.log`, e nada mais muda.
- Motivo: rigor (seção 7). Separar congelamento de abertura garante que o commit trava as escolhas.
- Alternativa descartada: 03 abrir o teste direto sem flag; menos auditável.

## 2026-10-02 — Resultado do teste (2023+), modelos congelados
- Fato (arquivos `reports/v2/final/metricas_teste.csv` e `antecedencia_teste.csv`):
  - Regressão do volume: o XGBRes SUPERA o B3 no teste, skill_B3 = +0,240 (h30), +0,229 (h60),
    +0,167 (h90). MAE XGBRes 2,24 / 3,75 / 5,22 vs B3 2,95 / 4,87 / 6,27.
  - PR-AUC de entrada (limiar 40): B3 0,98 / 0,98 / 0,98; XGBRes 0,97 / 0,91 / 0,95 (ambos altos).
  - Crise de 2025-26 (episódio iniciando 2025-08-06, limiar 40): o XGBRes antecipa o alerta
    +6 d (h30: 25 vs 19), +8 d (h60: 48 vs 40) e +11 d (h90: 75 vs 64) sobre o B3, com 0 falso-alarme
    no teste (exceto h60, 2 falso-alarmes do XGBRes).
- Interpretação (muda a narrativa, relatado ao Victor): na VALIDAÇÃO (2003-2022, 4 folds) o XGBRes
  EMPATAVA com o B3; no TESTE (2023-2026, período único com a forte depleção de 2025-26) ele SUPERA o
  B3 e antecipa mais. Não é contradição do plano: a validação robusta (20 anos) mostrou empate; o teste
  (um período) mostrou vantagem na crise recente. Caveat honesto: é uma única realização de teste; não
  reabrir nem re-selecionar nada (teste já gasto). H1/H2 recebem apoio parcial no teste.
- Enquadramento aprovado pelo Victor (2026-10-02): a evidência principal é a validação (20 anos, 4
  folds, 10 episódios); o teste é realização única e confirma a direção.

## 2026-10-02 — Diagnósticos pós-hoc do teste (03b; NÃO altera o modelo congelado)
- Arquivo: `reports/v2/final/teste_diagnostico.csv`. Todos os números são pós-hoc.
- IC95% do skill vs B3 no teste (bootstrap em blocos de 90 d): h30 = 0,240 [0,132; 0,372] e
  h60 = 0,229 [0,079; 0,420] são SIGNIFICATIVOS (IC acima de zero); h90 = 0,167 [-0,052; 0,378]
  NÃO é significativo (cruza zero). Ou seja, a vantagem do XGBRes no teste se sustenta em 30 e 60 dias.
- Skill por período: a vantagem aparece tanto em 2023-2024 (calmo) quanto em 2025-2026 (crise):
  h30 0,242 vs 0,237; h60 0,258 vs 0,188; h90 0,164 vs 0,170. NÃO é artefato de um único evento.
- Diagnóstico do B3: o B3 congelado usa a variação sazonal de 1984-2022, misturando regras de operação
  pré e pós-Resolução 925 (2017). Um B3 ajustado SÓ em 2017-2022 melhora um pouco (MAE menor), mas
  explica apenas 12% a 25% da vantagem do XGBRes (h30 25%, h60 22%, h90 12%). Conclusão: a maior parte
  do ganho do XGBRes no teste é real, não é só efeito do baseline sazonal antigo. (Diagnóstico; o B3
  congelado continua sendo o oficial do teste.)
- Antecedência e falso-alarme no teste (limiares 40 e 30, inclui o episódio jun-set/2026) também em
  teste_diagnostico.csv.

## 2026-10-02 — Exp 4 (item 5, H4): o mesmo pipeline em outros reservatorios (time-boxed)
- Decisão: demonstrar H4 (portabilidade) rodando o pipeline nos 4 reservatorios INDIVIDUAIS do
  Cantareira (Jaguari-Jacarei 808 hm3, Cachoeira 70, Atibainha 96, Paiva Castro 8), que sao
  reservatorios distintos com series e climas proprios ja baixados (sem download novo).
- Motivo (time-box, 1 dia): os reservatorios FORA do Cantareira (Nordeste `/sar0/Medicao` e SIN
  `/sar0/MedicaoSin`) servem a serie por AJAX com estado de sessao; o HTML estatico traz so o
  cabecalho (0 <td>), ao contrario do `/sar0/MedicaoCantareira`, que e renderizado no servidor.
  `ExportarExcel` e stateful (erro 500 "source null") e `GraficoVolume` tambem e AJAX. Caminhos para
  depois (registrados): endpoint AJAX interno, Claude-in-Chrome dirigindo a pagina, ou dados abertos
  da ONS. O harness `exp4_outros_reservatorios.py` ja aceita qualquer reservatorio; falta so a coleta.
- Resultado (reports/v2/exp4/exp4_skill.csv), skill vs B3 na validacao (h30/60/90):
  - Jaguari-Jacarei: +0,113 / +0,049 / +0,005 (o maior; positivo, como o sistema agregado).
  - Cachoeira: -0,032 / +0,043 / +0,006 (empate).
  - Atibainha: -0,409 / -0,431 / -0,300 (pior que o B3).
  - Paiva Castro: -0,181 / -0,080 / -0,210 (pior que o B3).
- Interpretacao honesta: o pipeline e portavel (roda inalterado em 4 reservatorios), mas a utilidade
  depende da dinamica: empata/supera o B3 nos maiores e mais estaveis, e fica abaixo nos pequenos e
  volateis. Nota de dado: o "volume util %" dos pequenos vai a valores extremos (Atibainha min -106,8%),
  o que torna a serie ruidosa e dificil; vale tratar/entender esse % antes de usar esses reservatorios.
- ENCERRAMENTO (decisao do Victor, 2026-10-02): o Exp 4 fica encerrado como esta. Rodar o pipeline em
  reservatorios FORA do Cantareira (Nordeste e SIN do SAR, ou outros sistemas da Grande SP) fica como
  TRABALHO FUTURO: a coleta depende de acessar os endpoints AJAX/sessao do SAR (via endpoint interno,
  Claude-in-Chrome ou dados abertos da ONS). O harness exp4_outros_reservatorios.py ja aceita qualquer
  reservatorio; falta so a coleta.

## 2026-10-02 — Reprodutibilidade e regra de git (item 12, fechamento)
- Decisao do Victor: o item 12 fica encerrado pela reproducao 33/33 (venv limpo a partir do
  requirements.txt reproduz exatamente as configs, thresholds e metricas do teste). Nao relancar o
  run_all. A reproducao foi registrada em reports/v2/final/teste_aberto.log como REPRODUCAO do
  resultado congelado, sem nova abertura do teste (o modelo congelado nao muda).
- Estado do repo (herdado, pre-existente): o HEAD do Gustavo versiona a v1 sob `algoritmos/` e
  `resultados_*`, com um working tree divergente (v1 como `pipeline/`, `data/`, `models/`). Isso nao foi
  causado pela v2 e nao sera tocado (v1 congelada; reconciliar ou remover dados pesados reescreveria
  historico, decisao do Victor com o Gustavo).
- REGRA DE GIT ate o merge: nunca `git add -A` nem `git commit -a`; so `git add` com caminhos
  explicitos da v2. Evita arrastar a divergencia da v1 para os commits. (Registrada tambem no CLAUDE.md
  secao 6.)

## 2026-10-02 — Atibainha: explicacao do -106,8% e leitura da cascata (item 14)
- Fato (data/raw_v2/sar_cantareira_diario.csv): o minimo de volume util do Atibainha e -106,8% em
  2014-12-29, correspondente a volume util de -102,81 hm3 (capacidade util ~100 hm3). Nao e artefato de
  parsing: durante a crise de 2014-2015 o reservatorio foi puxado cerca de 103 hm3 ABAIXO do minimo util
  (uso profundo da reserva tecnica/volume morto). Foram 296 dias com volume negativo, 184 abaixo de -50%.
- Teste de massa (deltaV diario contra afluencia menos defluencia do SAR): nos TRES reservatorios
  testados a correlacao e so 0,49 (Jaguari), 0,54 (Atibainha) e 0,40 (Paiva Castro), e o residuo nao
  explicado tem magnitude parecida com a da propria variacao (residuo/|deltaV| ~ 1,0 a 1,6). Logo, a
  AFLUENCIA E A DEFLUENCIA DO SAR NAO INCLUEM AS VAZOES DOS TUNEIS entre reservatorios: elas capturam a
  afluencia natural e a liberacao controlada pelo rio, nao as transferencias operacionais (tuneis e
  bombeamento) que dominam a operacao do Cantareira.
- Conclusao: a leitura de que o resultado ruim do Atibainha vem da CASCATA SE SUSTENTA. O volume do
  Atibainha (e dos reservatorios pequenos/jusante) e governado pelas transferencias por tunel, que sao
  nao medidas e dirigidas por politica, nao pela chuva local. As features do modelo (clima local, vol em
  t, e a afluencia/defluencia PARCIAIS) nao capturam o driver dominante, entao o modelo fica abaixo ate
  do B3. Isso tambem explica por que o agregado do SISTEMA funciona melhor (Exp 2/3): no sistema as
  transferencias entre os 4 reservatorios sao internas e se cancelam, sobrando o balanco liquido do
  sistema. Implicacao para o texto e para trabalho futuro: para modelar reservatorios individuais do
  Cantareira seria preciso obter as vazoes de tunel/transferencia (dados de operacao da Sabesp/ANA).
