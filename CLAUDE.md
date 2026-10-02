# TG2 Fatec: previsão de crises hídricas no Sistema Cantareira

Leia este arquivo inteiro antes de qualquer tarefa. Ele vale mais do que qualquer instrução antiga no README ou no código da v1.

## 1. Contexto

- Trabalho de Graduação do curso de Ciência de Dados da Fatec Santana de Parnaíba.
- Autores: Victor Ribeiro Cunha e Gustavo Henrique Moises Martins. Orientador: Prof. Dr. Inacio Henrique Yano.
- Título, mantido do TG1: "Previsão de crises hídricas em São Paulo por meio da inferência de dados".
- O TG1 foi aprovado. O TG2 não foi apresentado no semestre passado porque a v1 tinha defeitos de construção: estações do INMET empilhadas sem identificador e com chuva somada, rótulo de crise manual que punia o modelo por antecipar, validação cruzada sem positivos e só um evento no teste. O diagnóstico completo está em `RELATORIO_ESTADO_ATUAL.md`.
- O orientador cobrou postura diante do resultado: testar outros dados, usar um dataset fictício para provar que o pipeline funciona, sair do F1 zero e manter o vínculo com o TG1.
- Entrega do texto em novembro de 2026 (data oficial a confirmar). Meta interna: versão final pronta em 9 de novembro.
- Plano de ação aprovado pelo Victor: https://claude.ai/code/artifact/f49912c0-5220-4a19-bb5e-22afc7caa1a9

## 2. Pergunta, hipóteses e alvo

Pergunta: variáveis climáticas, sozinhas ou somadas ao estado do reservatório, antecipam em 30, 60 e 90 dias a entrada do Sistema Cantareira em faixa de crise?

- H1: com dados corrigidos e alvo objetivo, os modelos superam persistência e climatologia.
- H2: clima sozinho tem poder preditivo limitado; o estado do reservatório agrega ganho mensurável.
- H3: em dados sintéticos com sinal conhecido, o pipeline recupera o sinal.
- H4: o mesmo pipeline funciona em outros reservatórios.

Alvo: volume útil do sistema (SAR/ANA) em t+30, t+60 e t+90. Crise definida pela Resolução Conjunta ANA/DAEE nº 925/2017:

- Principal: volume abaixo de 40% (Faixa 3, Alerta). São 17 episódios entre 1984 e 2026.
- Grave: volume abaixo de 30% (Faixas 4 e 5). São 5 episódios: 1986, 2003-04, 2013-16, 2021-22 e 2025-26.

## 3. Estado atual (2 de outubro de 2026)

- `pipeline_v1/` está congelado. Não alterar: ele é a base do Exp 0.
- `pipeline_v2/`:
  - `00_download_dados.py`: SAR desde 1984, ERA5-Land via Open-Meteo, NASA POWER e Sabesp, com `_manifest.json`.
  - `01_build_dataset_v2.py` e `dataset_v2.py`: dataset diário, features causais e episódios automáticos.
  - `transformadores.py`: SPI, SPEI e anomalias com fit no treino e transform no resto.
  - `02_treino_v2.py` e `modelagem_v2.py`: Exp 2, validação com janela crescente em 4 folds e purga.
  - `02b_analise_exp3.py` e `analise_exp3.py`: Exp 3, modelo residual sobre o B3 e skill condicional.
  - `02c_lstm.py` e `lstm_v2.py`: LSTM. `02d_antecedencia_pareada.py`: teste pareado da antecedência.
  - `04_experimento_sintetico.py` e `sintetico_v2.py`: Exp 1, sintético.
  - `config.yaml` e `tests/` (24 testes verdes).
- Resultados até aqui, todos na validação; o teste de 2023 em diante segue intocado:
  - Clima sozinho não prevê o volume (skill de −0,7 a −4,3 contra a persistência).
  - B3 (persistência mais variação sazonal) tem skill de +0,31 a +0,38 contra a persistência simples. É a régua.
  - XGBRes e LSTM empatam com o B3 no erro, inclusive na seca e no pré-episódio (todos os IC95% cruzam zero).
  - No limiar de 40%, o XGBRes antecipa o alerta em +19 dias com h = 60 (Wilcoxon p = 0,002) e em +6,5 dias com h = 30 (p = 0,039). Custo: 3 a 4 alarmes falsos em 20 anos, contra zero do B3.
  - A chuva da ERA5 é melhor que a do POWER; o POWER tem uma anomalia em 1999.
  - Sintético: o pipeline recupera o sinal, zera com o alvo embaralhado e atinge o teto no oráculo. Correção em andamento: medir a persistência efetiva (φ_ef) e rodar o gêmeo calibrado, porque as secas injetadas criavam persistência mesmo com φ = 0.

## 4. Backlog, em ordem

Execute em sequência, sem pedir confirmação entre os itens.

1. Sintético: φ_ef, cenários com N = 0 e gêmeo calibrado (prompt já enviado).
2. Versionamento: branch `tg2-v2` e commits por etapa (regras na seção 6).
3. `03_congelar_e_testar.py`:
   - grava em `reports/v2/final/congelamento.json` os conjuntos, modelos, hiperparâmetros e thresholds escolhidos na validação (B3 e XGBRes, limiares 40 e 30);
   - faz commit desse arquivo;
   - só então abre o teste uma única vez, com `--abrir-teste` e registro em `teste_aberto.log`;
   - relatório do teste com destaque para a crise de 2025-26: antecedência de cada modelo, alarmes falsos e skill contra B3.
4. Exp 0: figuras do diagnóstico da v1 a partir do relatório de auditoria: chuva anual contra número de estações, RAD_SUM zerada no teste, folds sem positivo e regra SPI contra os modelos.
5. Exp 4: o mesmo pipeline em outros sistemas da Grande SP (Sabesp) e em reservatórios do Nordeste (SAR). Limite de um dia de trabalho.
6. App Streamlit v2:
   - volume atual e previsão em 30, 60 e 90 dias com faixa de incerteza;
   - faixa ANA/DAEE prevista;
   - página de backtest (2003-04, 2013-16, 2021 e 2025-26);
   - página do sintético com selo de dado sintético;
   - atualização por API;
   - remover a menção à estação A701.
7. Material para o texto:
   - figuras e tabelas finais em `reports/v2/final/`;
   - um resumo em Markdown por experimento (`reports/v2/final/resumo_expN.md`), com números, figuras e interpretação;
   - o dicionário de dados.
8. Higiene: `run_all.py`, README da v2 e remoção do `.venv` do índice do git.

## 5. Autonomia

- Você tem autonomia para executar o backlog: decidir questões técnicas, criar e alterar arquivos da v2, rodar, testar, commitar e seguir para o próximo item. O Victor está em outras frentes e aprovou o plano.
- Pare e pergunte apenas quando:
  1. a decisão muda a pergunta de pesquisa, o alvo, os limiares de crise ou os critérios de sucesso;
  2. um resultado contradiz o plano a ponto de mudar a narrativa do TG (relate e proponha o caminho);
  3. a ação é irreversível fora do repositório local: push na main, merge na main, apagar dados brutos, reescrever histórico.
- Registre toda decisão relevante em `DECISOES.md`, com data, decisão, motivo e alternativa descartada. Esse arquivo vira insumo do texto.
- Ao fim de cada item do backlog, reporte em poucas linhas o que foi feito, os números principais, os arquivos, o resultado dos testes e o hash do commit. Depois siga para o próximo item.

## 6. Versionamento (regras automáticas, sem pedir aprovação)

- Trabalhe na branch `tg2-v2`. Faça commit ao fim de cada etapa concluída com testes verdes, com mensagem em português que descreva a etapa.
- Antes de cada commit, verifique automaticamente:
  - não entram `data/raw_v2/`, `data/processed_v2/`, `data/sintetico/`, `_auditoria_tmp/`, `.venv/` nem `.env`;
  - nenhum arquivo acima de 5 MB;
  - os testes passam.
  Se algo falhar, corrija o `.gitignore` ou o código e siga.
- Push permitido apenas na branch `tg2-v2`. Nunca force push, nunca reescrever histórico, nunca commitar ou fazer merge na main. O merge é decisão do Victor com o Gustavo.

## 7. Rigor científico (inegociável)

- O teste (de 2023-01-01 em diante) só é aberto uma vez, pelo 03, depois do congelamento. Depois de aberto, nenhuma escolha de modelo, feature ou threshold muda.
- Nenhuma estatística calculada sobre a série inteira fora de `transformadores.py` com fit no treino. Purga de h dias entre treino e validação.
- Baselines B1, B2 e B3 sempre reportados. O skill contra o B3 é a régua.
- Resultado negativo é reportado como é. Nunca ajustar análise para o número ficar bonito.
- Dado sintético sempre rotulado "DADOS SINTÉTICOS", em caminhos com "sintetico", nunca misturado com o real.
- Todo número citado vem de um arquivo gerado pelo pipeline. Inferência sua é marcada como inferência.
- Fontes e datas de acesso registradas no `_manifest.json`.

## 8. Convenções

- Python. Português do Brasil em código, comentários, commits e documentos. Sem travessão e sem emoji.
- Lógica em módulos importáveis (`dataset_v2.py`, `modelagem_v2.py` etc.) e orquestradores numerados. Parâmetros no `config.yaml`. Testes em `pipeline_v2/tests/`. Todo bug corrigido ganha o teste que o pega.
- O `.venv` versionado aponta para a máquina do Gustavo e não roda. Use um venv local criado a partir do `requirements.txt`, com dependências fixadas.
- Matplotlib com backend Agg. Figuras com título que diz o achado, unidades nos eixos e legíveis impressas.

## 9. Referências

- `RELATORIO_ESTADO_ATUAL.md`: auditoria da v1.
- Plano de ação: link na seção 1.
- TG de referência ANIMA (Fatec, 2025): o modelo de narrativa é a investigação contada passo a passo, inclusive o que falhou.