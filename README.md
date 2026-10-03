# Previsao de crises hidricas no Sistema Cantareira (TG2)

Trabalho de Graduacao do curso de Ciencia de Dados da Fatec Santana de Parnaiba. Autores: Victor
Ribeiro Cunha e Gustavo Henrique Moises Martins. Orientador: Prof. Dr. Inacio Henrique Yano. O objetivo
e prever, com 30, 60 e 90 dias de antecedencia, a entrada do Sistema Cantareira em faixa de crise, a
partir de dados climaticos e do estado dos reservatorios. O alvo e o volume util do sistema (SAR/ANA) e
a crise segue a Resolucao Conjunta ANA/DAEE 925/2017 (abaixo de 40% e Alerta; abaixo de 30% e grave).

Esta e a versao 2 do trabalho. A versao 1 tinha defeitos de construcao (estacoes do INMET empilhadas
sem identificador e com chuva somada, rotulo de crise manual, validacao cruzada sem positivos). A v2
reconstroi tudo: alvo objetivo, dados por fonte aberta (SAR, ERA5-Land, NASA POWER), validacao com
janela crescente e purga, baselines sempre reportados e um teste que so abre uma vez, depois do
congelamento das escolhas. O diagnostico da v1 esta em docs/RELATORIO_ESTADO_ATUAL.md.

## Resultado principal
- O estado do reservatorio e o preditor decisivo; o clima agregado mais ML empata com o baseline forte
  B3 (persistencia mais variacao sazonal) na validacao de 20 anos.
- No teste intocado (2023 em diante, modelo congelado), o modelo supera o B3 (skill 0,167 a 0,240) e
  antecipa a crise de 2025-26 em ate 11 dias.
- Resultado negativo e reportado como e; o teste foi aberto uma unica vez, apos o congelamento.

## Como instalar
Python 3.13. O `.venv` versionado antigo foi removido (apontava para outra maquina); crie o seu a partir do `requirements.txt`.

```
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
```

`requirements.txt` lista as dependencias diretas com as versoes exatas que geraram os resultados.
`requirements-lock.txt` e o congelamento completo do ambiente (todas as transitivas, 67 pacotes), para
reproducao bit a bit. Nao sao redundantes: o primeiro e para instalar, o segundo e para auditar.

## Como rodar
```
python pipeline_v2/00_download_dados.py --all     # 1. coleta SAR + ERA5 + NASA POWER (data/raw_v2/)
python pipeline_v2/01_build_dataset_v2.py         # 2. dataset diario, features e episodios
python pipeline_v2/run_all.py                     # 3. experimentos de validacao (Exp 0/1/2/3/4 + LSTM)
# congelamento e teste (o teste abre uma unica vez):
python pipeline_v2/03_congelar_e_testar.py        #    grava reports_v2/final/congelamento.json
python pipeline_v2/03_congelar_e_testar.py --abrir-teste
# aplicacao web:
.venv\Scripts\python.exe -m streamlit run app_v2\Home.py
# testes:
python -m pytest pipeline_v2/tests -q
```

## Estrutura de pastas
```
.
├── pipeline_v2/        codigo da v2 (coleta, dataset, modelos, experimentos, testes)
│   ├── 00_download_dados.py, 01_build_dataset_v2.py, 02_*, 03_*, 04_*, exp0/exp4, run_all.py
│   ├── dataset_v2.py, modelagem_v2.py, analise_exp3.py, lstm_v2.py, sintetico_v2.py, transformadores.py
│   ├── config.yaml
│   └── tests/          26 testes (pytest)
├── app_v2/             aplicacao Streamlit v2 (previsao, backtest, sintetico, sobre)
├── reports/
│   └── v2/             saidas da v2: eda/, resultados/, sintetico/, exp0/, exp4/, final/
│       └── final/      congelamento, metricas do teste, ficha tecnica, rascunho do capitulo 4,
│                       figuras finais (300 dpi) em final/figuras/ com indice_figuras.md
├── legado_v1/          versao do TG1, congelada, base do Exp 0 (ver legado_v1/README.md)
├── docs/               RELATORIO_ESTADO_ATUAL.md (auditoria da v1)
├── data/               dados locais, fora do git (raw_v2/, processed_v2/, sintetico/)
├── README.md, CLAUDE.md, DECISOES.md, requirements.txt, requirements-lock.txt, .gitignore
```

## Onde ficam figuras e relatorios
- Figuras finais do capitulo 4: `reports_v2/final/figuras/` (300 dpi) com `indice_figuras.md`.
- Rascunho do capitulo 4: `reports_v2/final/capitulo4_rascunho.md`. Valores da metodologia:
  `reports_v2/final/valores_capitulo3.md` e `ficha_tecnica.md`.
- Resultados por experimento: `reports_v2/resultados/`, `reports_v2/sintetico/`, `reports_v2/eda/`,
  `reports_v2/exp0/`, `reports_v2/exp4/`.
- Decisoes de projeto: `DECISOES.md`. Regras e backlog: `CLAUDE.md`.

## O que e o legado_v1/
Versao do TG1, congelada. Serve de base do Exp 0 (o diagnostico de por que a v1 falhou). Nao e
desenvolvida; so lida em modo leitura. Os dados pesados do INMET ficam fora do git; veja
`legado_v1/README.md` para a fonte e como regenerar.
