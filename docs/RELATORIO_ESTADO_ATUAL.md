# RELATÓRIO DO ESTADO ATUAL — Previsão de Crises Hídricas (TG)

> Auditoria técnica somente-leitura do repositório `code_tg`.
> Data da auditoria: 2026-10-01. Nenhum arquivo do projeto foi alterado, movido ou apagado.
> Reproduções e ambiente temporário em `_auditoria_tmp/`.
>
> **Convenções de evidência:**
> - `[FATO]` — verificado por leitura de código/arquivo ou execução; o caminho de origem é citado.
> - `[INFERÊNCIA]` — conclusão minha a partir dos fatos.
> - `[NÃO VERIFICADO]` — não foi possível confirmar; o motivo é indicado.
>
> **Ambiente usado na reprodução:** o `.venv` versionado aponta para `C:\Users\Gustavo\anaconda3`
> (outra máquina) e é cp39 — inutilizável aqui. Criei `_auditoria_tmp/venv` (Python 3.13.14) e
> instalei pandas 3.0.6, pyarrow, openpyxl, scikit-learn, xgboost, matplotlib, seaborn, streamlit,
> plotly. **Atenção:** as versões diferem das fixadas em `requirements.txt` (pandas 2.3.3,
> scikit-learn 1.6.1, xgboost 2.0.3). Por isso métricas reproduzidas por mim diferem levemente dos
> CSVs salvos no repositório (ver seção 8). Os CSVs salvos são a referência "oficial".

---

## 1. Resumo executivo

1. **O que existe:** pipeline completo em 4 scripts (`pipeline/01..04`), app Streamlit de 4 páginas,
   22 XLSX brutos do INMET (2003–2024), parquets intermediários/finais, 3 modelos `.pkl` e relatórios.
   O código roda. `[FATO]`
2. **O que roda:** passos 02 (features), 03 (rótulos) e 04 (treino) foram reproduzidos em
   `_auditoria_tmp/` e batem com os artefatos existentes (rótulos 45/6957; AUC y30 na mesma faixa). `[FATO]`
3. **O que não roda sem ajuste:** o `.venv` do repo está quebrado (interpretador de outra máquina);
   o passo 04 **quebra em ambiente headless** porque usa o backend Tk do matplotlib
   (`RuntimeError: main thread is not in main loop`) — só completa com `MPLBACKEND=Agg`. `[FATO]`
4. **5 problemas mais graves:**
   1. **Os dados NÃO são de uma estação.** Cada ano empilha de ~4 (2003) a ~43 (2020) estações do
      estado **sem identificador**, e a precipitação é **somada entre estações**. O "total anual de
      chuva" salta de 3.438 mm (2003) para 50.475 mm (2019), acompanhando o número de estações, não o
      clima. O README/app afirmam "Estação São Paulo – Mirante (A701)", o que é **falso**. `[FATO]`
   2. **`RAD_SUM` = 0 em todo o teste (2020–2024).** A coluna de radiação muda de grafia
      (`(KJ/m²)`→`(Kj/m²)`) a partir de 2020 e a agregação só lê a primeira; a radiação vira zero em
      todo o período de teste e fica inflada/crescente no treino. `[FATO]`
   3. **A validação cruzada é degenerada.** Em 4 dos 5 folds do `TimeSeriesSplit` há **zero
      positivos**; o F1 médio é 0 e o "threshold ótimo" cai sempre em 0,01 (mínimo da grade). Os três
      modelos em produção têm threshold = 0,01. `[FATO]`
   4. **Desempenho real ≈ aleatório / pior que baseline.** No teste, F1 ≈ 0, PR-AUC ≈ 0,01–0,02
      (prevalência = 0,0107). Uma regra univariada `-SPI30` atinge AUC 0,75–0,78, **melhor que todos
      os modelos**. `[FATO]`
   5. **Só há 1 evento de crise no teste** (2021) e 2 no treino. Com janelas de 15 dias, são 45
      "positivos" que na prática representam **3 eventos** altamente autocorrelacionados — base
      estatística insuficiente. Além disso, o SPI usa climatologia de toda a série (vazamento). `[FATO]`

---

## 2. Inventário

### 2.1 Árvore de diretórios (3 níveis) e volume dos dados `[FATO]`

```
code_tg/                               (.git = 712 MB — ver seção 13)
├── TRABALHO ... (1).pdf               (em C:\Dev\TG, fora do repo; 1,18 MB — documento do TG)
├── .git/                              712 MB  (dados pesados versionados)
├── .gitignore
├── .venv/                             (quebrado: aponta p/ C:\Users\Gustavo\anaconda3, cp39)
├── README.md
├── requirements.txt
├── data/
│   ├── raw/        439 MB   22×*.xlsx (2003_base.xlsx 2,5 MB … 2024_base.xlsx 29 MB)
│   ├── interim/     97 MB   inmet_sp_hourly_clean.parquet (96 MB) + qc_hourly.csv
│   ├── features/   2,6 MB   inmet_sp_daily.parquet (450 KB), _features (1,07 MB),
│   │                        _labels (1,07 MB), _labels_stats.csv
│   └── events/      1 KB    eventos.csv
├── pipeline/
│   ├── 01_clean_hourly.py
│   ├── 02_build_features.py
│   ├── 03_build_labels.py
│   └── 04_train_model.py
├── models/         692 KB
│   ├── v1_threshold_padrao/        modelo_crise_hidrica_y90.pkl (4 KB)
│   └── v2_threshold_otimizado/     y30.pkl (682 KB, RandomForest), y60.pkl (4 KB, DT),
│                                   y90.pkl (4 KB, DT)   ← usado pelo app
├── reports/        1,4 MB
│   ├── v1_threshold_padrao/        1 CSV + 6 PNG (só y90)
│   └── v2_threshold_otimizado/     3 CSV + 27 PNG (y30/y60/y90)
└── app/
    ├── Home.py
    ├── utils.py
    └── pages/   1_Previsao.py  2_Visualizacao.py  3_Sobre.py
```

### 2.2 Scripts, módulos e o que cada um faz `[FATO]`

| Arquivo | Função (1 linha) |
|---|---|
| `pipeline/01_clean_hourly.py` | Lê os 22 XLSX, normaliza data/hora, converte vírgula→ponto, trata `-9999`, concatena tudo num parquet horário + tabela de QC. |
| `pipeline/02_build_features.py` | Agrega hora→dia (precip=soma, temp=média…) e cria 22 features (P7/15/30/90, dry spells, médias móveis, SPI30 aproximado). |
| `pipeline/03_build_labels.py` | Marca `y30/y60/y90`=1 nas janelas de 15 dias antes de cada evento; exclui (NaN) os dias durante a crise. |
| `pipeline/04_train_model.py` | Treina DecisionTree/RandomForest/XGBoost, "otimiza" threshold por `TimeSeriesSplit`, avalia no teste, salva o melhor `.pkl` + gráficos + CSV. |
| `app/Home.py` | Página inicial: índice de risco do período (prob. média normalizada pelo threshold). |
| `app/pages/1_Previsao.py` | Previsão por upload de CSV ou inserção manual de 1 dia. |
| `app/pages/2_Visualizacao.py` | Séries temporais históricas com crises destacadas. |
| `app/pages/3_Sobre.py` | Texto descritivo do projeto, metodologia e limitações. |
| `app/utils.py` | Carrega modelo/histórico, lista `FEATURES` (22), valida CSV, faz a previsão. |

Não há notebooks (`.ipynb`) no repositório. `[FATO]`

### 2.3 Dependências `[FATO]` (`requirements.txt`)

Todas **fixadas** (`==`). Python declarado: README "3.9+"; `.venv/pyvenv.cfg` = 3.9.13.
pandas 2.3.3 · numpy 2.0.2 · pyarrow 21.0.0 · openpyxl 3.1.5 · scikit-learn 1.6.1 ·
xgboost 2.0.3 · joblib 1.5.3 · matplotlib 3.9.4 · seaborn 0.13.2 · streamlit 1.50.0 · plotly 6.7.0.
Não há `pyproject.toml`, `environment.yml` nem lockfile de pip.

### 2.4 Git `[FATO]`

- **Commits:** 5. **Autor (único):** Gustavo.
- **Primeiro:** 2025-11-19. **Último:** 2026-06-02.
- Lista completa (há apenas 5, não 15):

```
624f225d | 2026-06-02 | Gustavo | Construção Streamlit e Treinamento
4c71e669 | 2026-05-22 | Gustavo | Mudanças - correções nos algoritmos e execução de modelo
22fe006f | 2025-11-19 | Gustavo | Mapeamento de eventos
ebae9526 | 2025-11-19 | Gustavo | Até as Features
d788a8ba | 2025-11-19 | Gustavo | first commit
```

### 2.5 Documentação existente `[FATO]`

- **`README.md`** — descreve estrutura, instalação, ordem do pipeline (com os comandos exatos),
  o app e a modelagem. Afirma "INMET, Estação São Paulo – Mirante (A701), 2003–2024".
- **`app/pages/3_Sobre.py`** — repete metodologia, lista as 22 features, os 3 eventos, o split
  80/20 (treino 2003–2020 / teste 2021–2024) e **reconhece** limitações: só 3 eventos, ausência de
  variáveis hidrológicas, não generaliza para outras regiões.
- Não há `CONTRIBUTING`, `docs/`, nem docstrings de módulo além dos cabeçalhos dos scripts.

---

## 3. Como o pipeline roda hoje

### 3.1 Ordem real, entradas e saídas `[FATO]`

| # | Script | Entrada | Saída | Caminho padrão |
|---|---|---|---|---|
| 1 | `01_clean_hourly.py` | `data/raw/*.xlsx` | `inmet_sp_hourly_clean.parquet`, `qc_hourly.csv` | `data/interim/` |
| 2 | `02_build_features.py` | hourly parquet | `inmet_sp_daily.parquet`, `inmet_sp_daily_features.parquet` | `data/features/` |
| 3 | `03_build_labels.py` | features parquet + `eventos.csv` | `inmet_sp_daily_labels.parquet` + `_stats.csv` | `data/features/` |
| 4 | `04_train_model.py` | labels parquet | `.pkl` do melhor modelo + PNGs + `comparativo_*.csv` | `models/`, `reports/` |
| — | `app/Home.py` (+pages) | `models/v2_.../*.pkl` + labels parquet | dashboard | — |

### 3.2 Caminhos e parâmetros `[FATO]`

- **Sem arquivo de configuração.** Tudo é `argparse` com defaults relativos a
  `ROOT = Path(__file__).resolve().parents[1]`. Os defaults já apontam para `data/…`.
- **Parâmetro crítico obrigatório:** o passo 03 precisa de `--default-system Cantareira`
  (e `--exclude-during`); sem isso nenhum evento é rotulado (o README avisa). `[FATO]`
- Hiperparâmetros e `random_state=42` estão **hardcoded** em `04_train_model.py`.
- Caminhos do app são hardcoded em `app/utils.py` (`MODELS_DIR = .../v2_threshold_otimizado`).

### 3.3 Reprodução ponta a ponta (gravada em `_auditoria_tmp/repro_pipeline/`) `[FATO]`

| Passo | Resultado | Tempo | Observação |
|---|---|---|---|
| 01 | **não reexecutado** | — | Evitado: `.venv` quebrado + 22 XLSX (439 MB, o de 2024 tem 29 MB) são muito lentos com openpyxl; a saída (5,55 M linhas) foi **validada diretamente** no parquet existente. |
| 02 | OK, idêntico | ~7 s | gerou daily + features com 8.036 linhas × 26 colunas. |
| 03 | OK, idêntico | ~3 s | `labels_stats.csv` reproduzido = `y30/y60/y90: 45 / 6957 / 0,6427%`. |
| 04 (y30) | OK com `MPLBACKEND=Agg` | ~43 s | reproduziu `comparativo_modelos_y30` (ver 8). |

**Onde quebrou:** `04_train_model.py` sem `MPLBACKEND=Agg` lança
`RuntimeError: main thread is not in main loop` / `Tcl_AsyncDelete` no estágio de plotagem
(backend Tk em ambiente sem display). Causa: o script não fixa backend não-interativo do matplotlib.
Para rodar seria preciso `import matplotlib; matplotlib.use("Agg")` antes de `pyplot`. `[FATO]`

---

## 4. Dados brutos

### 4.1 Fontes, formato, período, quantidade `[FATO]`

- **Fonte:** INMET (BDMEP). **Formato:** 22 arquivos `AAAA_base.xlsx`, uma planilha `Sheet1` cada,
  20 colunas (data, hora UTC e 17 variáveis + 1 coluna vazia `Unnamed: 19`). Valores decimais com
  vírgula; ausentes como `-9999`.
- **Período:** 2003-01-01 00:00 → 2024-12-31 23:00 (UTC). **Volume bruto:** 439 MB.
- **Parquet horário unificado:** **5.548.008 linhas** × 24 colunas (`data/interim/…hourly_clean.parquet`).

### 4.2 Há identificação de estação? **NÃO — e isto é decisivo** `[FATO]`

Verifiquei os cabeçalhos dos **22** arquivos: todos têm as mesmas 20 colunas e **nenhuma**
identifica estação (código, nome, lat/lon, município). A heurística do próprio código
(`guess_station_col`) não acha nada; o único "match" textual é a palavra "ESTACAO" **dentro** do
nome da variável `PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO`.

**Porém os arquivos contêm MÚLTIPLAS estações empilhadas, sem rótulo.** O número de linhas por
timestamp (mesma data+hora) cresce ao longo do tempo:

| Ano | linhas | linhas por timestamp (≈ nº de estações) |
|---|---|---|
| 2003 | 34.224 | 3,9 |
| 2006 | 70.152 | 8,0 |
| 2008 | 240.144 | 27,3 |
| 2014 | 271.560 | 31,0 |
| 2020 | 377.712 | **43,0** |
| 2024 | 351.360 | 40,0 |

> **Conclusão:** a afirmação do documento de que "não havia identificador e por isso tudo virou
> média do estado de SP" está **CONFIRMADA quanto à ausência de identificador**, mas a consequência
> é mais grave do que "média": a composição de estações muda de ~4 para ~43 ao longo da série, e
> (ver 5.1) a precipitação e a radiação são **somadas** entre estações. A alegação de "Estação São
> Paulo – Mirante (A701)" no README e no app é **incorreta**. `[FATO]` + `[INFERÊNCIA]`

- **Lista de estações com código/nome/lat/lon e anos de cobertura:** `[NÃO VERIFICADO]` — impossível,
  pois os arquivos não trazem nenhum identificador para desempilhar as estações.
- **Alguma estação na região do Cantareira (PCJ)?** `[NÃO VERIFICADO]` — sem identificador, não há
  como saber quais municípios/estações compõem o agregado. `[INFERÊNCIA]`: é provável que o conjunto
  seja de estações automáticas do estado de SP (e não da bacia do Cantareira), mas não é verificável.

### 4.3 Taxa de ausentes por variável e por ano `[FATO]`

Nas features **diárias** agregadas, os ausentes são baixíssimos (a agregação mascara buracos — ver
5.2). Percentual de dias com NaN por variável-base:

| Ano | PRECIP | TMEAN | TMAX | TMIN | URMEAN | RAD_SUM | PRESSAO | VENTO | RAJADA |
|---|---|---|---|---|---|---|---|---|---|
| 2005 | 0,0 | 1,4 | 1,4 | 1,4 | 1,4 | 0,0 | 1,4 | 1,4 | 1,4 |
| 2006 | 0,0 | 0,5 | 0,5 | 0,5 | 0,5 | 0,0 | 0,8 | 1,4 | 1,4 |
| demais anos | 0,0 | 0,0 | 0,0 | 0,0 | 0,0 | 0,0 | 0,0 | 0,0 | 0,0 |

> **Ressalva importante:** "0,0% ausente" é **enganoso**. Como a agregação diária usa `sum`/`mean`
> sobre o conjunto de estações, um dia só fica NaN se **todas** as estações faltarem naquele dia.
> E `RAD_SUM` aparece como "0% ausente" mesmo sendo **zero** em 2020–2024 (ver 4.4). `[FATO]`

### 4.4 Defeito oculto: `RAD_SUM` = 0 em todo o teste `[FATO]`

Há **duas grafias** da coluna de radiação no parquet horário:
`RADIACAO GLOBAL (KJ/m²)` (preenchida 2003–2019) e `RADIACAO GLOBAL (Kj/m²)` (preenchida 2020–2024).
O passo 02 só agrega a primeira. Resultado:

| Período | `(KJ/m²)` preenchida | `(Kj/m²)` preenchida | `RAD_SUM` diário |
|---|---|---|---|
| 2003–2019 | ~20–55% | 0% | cresce 35.914 → 677.231 (inflado pelo nº de estações) |
| 2020–2024 | 0% | ~33–53% | **0 em 100% dos dias** |

Ou seja, a feature de radiação é **inflada e crescente no treino e exatamente zero no teste**.

### 4.5 Dados hidrológicos no repositório? **NÃO** `[FATO]`

Não existe nenhum dado de volume/vazão/nível de reservatório, ANA, Sabesp, NDVI ou SPI oficial no
repositório. O único insumo "hidrológico" é o `eventos.csv` (datas de crise anotadas à mão) e o
`SPI30_APRX` (derivado da precipitação agregada). O próprio `3_Sobre.py` reconhece: "variáveis
hidrológicas … não foram incluídas nesta versão". As features são 100% climáticas do INMET.

---

## 5. Limpeza e agregação

### 5.1 Agregação entre estações e hora→dia `[FATO]` (`pipeline/02_build_features.py`)

Não há coluna de estação detectável, então o ramo executado é `df.resample('D').agg(aggs)` —
agrega **globalmente por dia**, misturando todas as estações daquele dia. Funções por variável:

```python
aggs = {
    'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)'              : 'sum',   # PRECIP_DIARIA
    'TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)'  : 'mean',  # TMEAN
    'TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)'    : 'max',   # TMAX
    'TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)'    : 'min',   # TMIN
    'UMIDADE RELATIVA DO AR, HORARIA (%)'           : 'mean',  # URMEAN
    'RADIACAO GLOBAL (KJ/m²)'                       : 'sum',   # RAD_SUM  (só a grafia maiúscula!)
    'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, ...'  : 'mean',  # PRESSAO_MED
    'VENTO, VELOCIDADE HORARIA (m/s)'               : 'mean',  # VENTO_MED
    'VENTO, RAJADA MAXIMA (m/s)'                    : 'max',   # RAJADA_MAX
}
```

> **Consequência da soma entre estações:** `PRECIP_DIARIA` e `RAD_SUM` são **somados sobre ~4 a ~43
> estações**. Prova quantitativa (total anual de `PRECIP_DIARIA`, `data/features/inmet_sp_daily.parquet`):
>
> | Ano | chuva "anual" (mm) | média diária (mm) | nº estações |
> |---|---|---|---|
> | 2003 | 3.438 | 9,4 | ~4 |
> | 2014 | 29.178 | 79,9 | ~31 |
> | 2019 | 50.475 | 138,3 | ~41 |
> | 2021 | 19.432 | 53,2 | ~42 |
>
> Uma estação de SP recebe ~1.300–1.500 mm/ano. Os valores são 2–35× maiores e acompanham o nº de
> estações, não o clima. Isso contamina `PRECIP_DIARIA`, `P7/P15/P30/P90`, dry spells e `SPI30_APRX`,
> e cria uma **mudança de escala entre treino e teste** que nada tem a ver com crise hídrica. `[FATO]`
>
> Temperatura/umidade/pressão usam `mean`/`max`/`min` → viram média espacial de todo o estado,
> perdendo o sinal local. `[INFERÊNCIA]`

### 5.2 Tratamento de -9999, NaN, outliers, dias faltantes `[FATO]`

- **-9999:** convertido para NaN em `coerce_numeric` (`01_clean_hourly.py`, trata `'-9999'`, `'-9999.0'`,
  `'-9999,0'` e também o inteiro `-9999` via `astype(str)`).
- **Decimais:** vírgula→ponto; remoção de espaços; `to_numeric(errors='coerce')`.
- **Umidade relativa:** valores fora de [0,100] viram NaN (único tratamento de outlier).
- **Linhas sem timestamp válido:** descartadas.
- **Imputação:** **não há** imputação estatística. O NaN só é "preenchido" de duas formas:
  (a) na agregação diária, `sum`/`mean` ignoram NaN (e `sum` de tudo-NaN = **0**, confundindo
  "seco/zero" com "faltante" — ver RAD_SUM); (b) no treino, `X.fillna(0)` em `04_train_model.py`
  e no app — **todo NaN de feature vira 0**, inclusive SPI e dry spells.
- **Dias faltantes:** a série diária tem 8.036 dias **contínuos** (2003-01-01…2024-12-31, sem buracos).

---

## 6. Engenharia de features

### 6.1 Tabela das 22 features `[FATO]` (`02_build_features.py`; todas sobre a série diária agregada)

| Feature | Fórmula | Janela | Causal? |
|---|---|---|---|
| PRECIP_DIARIA | soma horária do dia (entre estações) | 1 dia | sim |
| TMEAN/TMAX/TMIN | mean/max/min horário do dia | 1 dia | sim |
| URMEAN | média horária | 1 dia | sim |
| RAD_SUM | soma horária (**só grafia `(KJ/m²)`**) | 1 dia | sim |
| PRESSAO_MED | média horária | 1 dia | sim |
| VENTO_MED / RAJADA_MAX | mean / max horário | 1 dia | sim |
| P7 / P15 / P30 / P90 | `PRECIP_DIARIA.rolling(w).sum()`, `min_periods=0.7w` | 7/15/30/90 d | **sim (passado)** |
| DRY_STREAK_CUR | nº de dias consecutivos com chuva < 1 mm | acumulada | sim |
| DRY30_MAX / DRY90_MAX | `DRY_STREAK_CUR.rolling(30/90).max()` | 30/90 d | **sim (passado)** |
| TMEAN_MA7 / TMEAN_MA30 | média móvel de TMEAN | 7/30 d | **sim (passado)** |
| URMEAN_MA7 / URMEAN_MA30 | média móvel de URMEAN | 7/30 d | **sim (passado)** |
| PRESSAO_MED_MA7 | média móvel de PRESSAO_MED | 7 d | **sim (passado)** |
| SPI30_APRX | z-score mensal de P30 (ver 6.2) | 30 d + climatologia | **NÃO (vaza)** |

Todas as janelas móveis usam `.rolling(w)` (trailing) → **causais**. A única exceção é o SPI30. `[FATO]`

### 6.2 SPI30_APRX usa estatísticas de toda a série (incl. teste) → **VAZAMENTO** `[FATO]`

```python
daily['mes'] = daily['timestamp'].dt.month
clim = (daily.dropna(subset=['P30'])
            .groupby(keys)['P30']                 # keys = ['mes']  (sem estação)
            .agg(['mean','std'])                  # média/desvio sobre TODA a série
            .rename(columns={'mean':'P30_MEAN_MES','std':'P30_STD_MES'})
            .reset_index())
daily = daily.merge(clim, on=keys, how='left')
daily['SPI30_APRX'] = (daily['P30'] - daily['P30_MEAN_MES']) / daily['P30_STD_MES']
```

A climatologia mensal (`mean`/`std` de P30 por mês) é calculada **sobre todos os anos, incluindo
2020–2024 (teste)**. Logo a feature de cada linha de treino "conhece" a distribuição do teste.
Vazamento confirmado. (Agrava-se porque P30 já está contaminado pela soma entre estações.) `[FATO]`

### 6.3 Outras features com informação futura `[FATO]`

Além do SPI30, nenhuma outra feature usa o futuro — `P30_MEAN_MES`/`P30_STD_MES` são intermediárias e
**excluídas** do treino (seção 8). As janelas móveis são todas trailing. `[FATO]`

---

## 7. Rotulagem

### 7.1 Conteúdo integral de `eventos.csv` `[FATO]`

```
sistema,start_date,end_date,criterio
Cantareira,2014-02-01,2015-10-31,Volume útil abaixo de 10% — crise hídrica de SP 2014-2015
Cantareira,2021-05-01,2021-11-30,Alerta de crise hídrica — estiagem prolongada 2021
Cantareira,2003-11-01,2004-04-30,Nível crítico do Sistema Cantareira 2003-2004
```

### 7.2 Lógica da rotulagem `[FATO]` (`03_build_labels.py`, `label_pre_evento`)

```python
for h in horizons:                       # (30, 60, 90)
    win_ini = t0_ev - pd.Timedelta(days=h + w_pre)   # w_pre = 7
    win_fim = t0_ev - pd.Timedelta(days=h - w_pos)   # w_pos = 7
    mask_time = (ts >= win_ini) & (ts <= win_fim)
    out.loc[mask_sys & mask_time, f'y{h}'] = 1        # janela de 15 dias
if exclude_during and pd.notna(t1_ev):
    out.loc[mask_sys & (ts>=t0_ev)&(ts<=t1_ev), [...]] = np.nan   # crise → NaN
```

> Confirma o documento: para cada horizonte `h`, marca `y_h=1` numa **janela de 15 dias**
> (`[t0-(h+7), t0-(h-7)]`) **centrada em h dias antes do início** da crise; os dias **durante** a
> crise viram **NaN** (removidos do treino). `[FATO]`

### 7.3 Positivos / negativos / NaN por horizonte `[FATO]`

(`data/features/inmet_sp_daily_labels.parquet`; split replicado conforme seção 8)

| Alvo | Total pos | Total neg | Total NaN | Treino pos | Teste pos |
|---|---|---|---|---|---|
| y30 | 45 | 6.957 | 1.034 | 30 | **15** |
| y60 | 45 | 6.957 | 1.034 | 30 | **15** |
| y90 | 45 | 6.957 | 1.034 | 30 | **15** |

Os 3 horizontes têm distribuição idêntica (3 eventos × 15 dias = 45). Os 1.034 NaN = soma das durações
das 3 crises. `[FATO]`

### 7.4 O critério de início das crises é objetivo? **Não — é anotação manual** `[FATO]`

As datas e os critérios estão **digitados à mão** em `eventos.csv`. Os textos citam limiares
("volume útil abaixo de 10%"), mas **não há** série de volume do Cantareira no projeto para derivar
essas datas automaticamente; são datas escolhidas pelo autor. `[FATO]` + `[INFERÊNCIA]` (não
reprodutíveis a partir de dado algum presente no repositório).

---

## 8. Treinamento e avaliação

### 8.1 Split e folds `[FATO]` (`04_train_model.py`)

- Carga: `dropna(subset=[alvo])` → **7.002 linhas** (remove os 1.034 dias de crise). `fillna(0)` nas features.
- Split cronológico 80/20 (sem shuffle):
  - **Treino:** 5.601 dias, **2003-01-01 → 2020-07-30**, 30 positivos.
  - **Teste:** 1.401 dias, **2020-07-31 → 2024-12-31**, 15 positivos.
- `TimeSeriesSplit(n_splits=5)` sobre o **treino** (5.601 dias → 5 folds de validação de 933 dias cada).
- **Positivos por fold de validação (reproduzido, igual p/ todos os modelos e horizontes):**

```
fold (n_val, pos_val) = [(933, 0), (933, 0), (933, 0), (933, 15), (933, 0)]
```

> **4 dos 5 folds têm ZERO positivos.** Só o fold 4 (que cobre o evento de 2014–2015) tem 15. `[FATO]`

### 8.2 Otimização de threshold e contato com o teste `[FATO]`

- O threshold é buscado numa grade `0,01…0,99` **só no treino**, maximizando o **F1 médio** nos 5
  folds do `TimeSeriesSplit`. O teste **não** é usado para escolher o threshold. `[FATO]`
- Como 4 folds têm 0 positivos, o F1 é 0 em quase toda a grade → o `argmax` escolhe o **primeiro**
  valor (0,01). Daí **todos** os modelos saírem com threshold = 0,01 e `F1 (CV) = 0`. `[FATO]`
- **Contato indireto com o teste:** a escolha do "melhor modelo" salvo usa o **F1 no teste**
  (`melhor = max(resultados, key=lambda r: r["f1"])`, linha 343). Logo a seleção de modelo olha o
  teste. Impacto prático pequeno (F1≈0 para quase todos), mas metodologicamente é seleção pelo teste. `[FATO]`

### 8.3 Hiperparâmetros, seeds, desbalanceamento `[FATO]`

| Modelo | Hiperparâmetros | Desbalanceamento | seed |
|---|---|---|---|
| DecisionTree | `max_depth=10` | `class_weight='balanced'` | 42 |
| RandomForest | `n_estimators=200, max_depth=15, n_jobs=-1` | `class_weight='balanced'` | 42 |
| XGBoost | `n_estimators=200, max_depth=8, learning_rate=0.1, eval_metric='logloss'` | `scale_pos_weight=185,70` | 42 |

`scale_pos_weight` = nº_neg/nº_pos no treino = 5.571/30 = **185,70** (reproduzido). `[FATO]`

### 8.4 Métricas — CSVs salvos no repositório (referência oficial) `[FATO]`

**v2 (threshold otimizado)** — `reports/v2_threshold_otimizado/comparativo_*.csv`:

| Alvo | Modelo | Threshold | F1 (teste) | AUC-ROC (teste) | F1 (CV) |
|---|---|---|---|---|---|
| y30 | DecisionTree | 0,01 | 0,0000 | 0,4888 | 0,0 |
| y30 | RandomForest | 0,01 | 0,0312 | **0,6819** | 0,0 |
| y30 | XGBoost | 0,01 | 0,0000 | **0,6995** | 0,0 |
| y60 | DecisionTree | 0,01 | 0,0 | 0,5000 | 0,0 |
| y60 | RandomForest | 0,01 | 0,0 | 0,4921 | 0,0 |
| y60 | XGBoost | 0,01 | 0,0 | 0,4182 | 0,0 |
| y90 | DecisionTree | 0,01 | 0,0 | 0,5000 | 0,0 |
| y90 | RandomForest | 0,01 | 0,0 | 0,4646 | 0,0 |
| y90 | XGBoost | 0,01 | 0,0 | 0,5918 | 0,0 |

**v1 (threshold padrão 0,5)** — só y90: DT AUC 0,5 / RF 0,4646 / XGB 0,5918; **F1 = 0 em todos**
(com threshold 0,5 nunca cruzado). `[FATO]`

### 8.5 Métricas — reprodução completa pela auditoria (`_auditoria_tmp/repro_train.py`) `[FATO]`

Inclui PR-AUC, precisão, recall e matriz de confusão (que os CSVs não trazem). **Nota:** libs mais
novas → AUC de RF/XGB diferem dos CSVs; DecisionTree bate exatamente.

| Alvo | Modelo | thr | AUC-ROC | PR-AUC | F1 | Precisão | Recall | Matriz [[TN,FP],[FN,TP]] |
|---|---|---|---|---|---|---|---|---|
| y30 | DecisionTree | 0,01 | 0,4888 | 0,011 | 0,000 | 0,00 | 0,00 | [[1355,31],[15,0]] |
| y30 | RandomForest | 0,01 | 0,6635 | 0,019 | 0,037 | 0,02 | 0,20 | [[1243,143],[12,3]] |
| y30 | XGBoost | 0,01 | **0,7442** | 0,020 | 0,000 | 0,00 | 0,00 | [[1357,29],[15,0]] |
| y60 | DecisionTree | 0,01 | 0,5000 | 0,011 | 0,000 | 0,00 | 0,00 | [[1386,0],[15,0]] |
| y60 | RandomForest | 0,01 | 0,4784 | 0,011 | 0,000 | 0,00 | 0,00 | [[1378,8],[15,0]] |
| y60 | XGBoost | 0,01 | 0,4556 | 0,012 | 0,000 | 0,00 | 0,00 | [[1386,0],[15,0]] |
| y90 | DecisionTree | 0,01 | 0,5000 | 0,011 | 0,000 | 0,00 | 0,00 | [[1386,0],[15,0]] |
| y90 | RandomForest | 0,01 | 0,4499 | 0,011 | 0,000 | 0,00 | 0,00 | [[1323,63],[15,0]] |
| y90 | XGBoost | 0,01 | 0,5838 | 0,017 | 0,000 | 0,00 | 0,00 | [[1386,0],[15,0]] |

> **Comparação com o documento (AUC 0,68–0,70 em 30 dias):** **CONFIRMADO** para y30 nos CSVs
> salvos (RF 0,6819; XGB 0,6995). Mas é a *única* métrica razoável: y60/y90 ficam em ~0,42–0,59
> (nível aleatório) e o **F1/precisão/recall são ~0 em praticamente tudo** — o modelo não acerta
> nenhum positivo de forma útil. `[FATO]`

### 8.6 Baselines (não existiam; calculados pela auditoria) `[FATO]`

Não há baseline no código. Calculei no mesmo split do teste:

| Baseline | AUC-ROC y30 | AUC-ROC y60 | AUC-ROC y90 | PR-AUC (≈) |
|---|---|---|---|---|
| Aleatório (= prevalência) | 0,50 | 0,50 | 0,50 | **0,0107** |
| Regra `-P30` (menos chuva → risco) | 0,596 | 0,452 | 0,362 | ~0,01 |
| Regra `-P90` | 0,448 | 0,291 | 0,354 | ~0,01 |
| Regra `-SPI30_APRX` | **0,785** | **0,755** | **0,752** | ~0,02 |

> **Resultado-chave:** uma regra de **um único limiar sobre o SPI30** supera, em AUC-ROC, **todos**
> os modelos treinados nos 3 horizontes. O PR-AUC de todos (modelos e baselines) é ~0,01–0,02,
> i.e., **mal distinguível do aleatório** (prevalência 0,0107). O pipeline de ML, como está, não
> agrega valor sobre uma regra trivial. `[FATO]` + `[INFERÊNCIA]`

### 8.7 Importância de features (top 10, reprodução) `[FATO]`

Dominam **PRESSAO_MED_MA7, DRY90_MAX, URMEAN_MA30, DRY30_MAX, TMIN**. Exemplos:
- y30 RandomForest: PRESSAO_MED_MA7 0,234 · PRESSAO_MED 0,115 · SPI30_APRX 0,107 · URMEAN_MA30 0,086 · P90 0,069 …
- y90 DecisionTree: DRY90_MAX 0,664 · URMEAN_MA30 0,214 · TMIN 0,071 …
- y90 XGBoost: DRY90_MAX 0,597 · URMEAN_MA30 0,115 · TMIN 0,111 …

As features de chuva acumulada (P7/P15/P30/P90) têm peso baixo; a pressão média móvel domina — o que
`[INFERÊNCIA]` sugere o modelo capturando sazonalidade/tendência, não um sinal de pré-crise.

### 8.8 Artefatos salvos `[FATO]`

- Modelos (`.pkl`, joblib, dict `{modelo, threshold}`), datados 2026-09-30 (checkout):
  `models/v2_threshold_otimizado/`: **y30=RandomForest (682 KB, thr 0,01)**, **y60=DecisionTree (4 KB)**,
  **y90=DecisionTree (4 KB)**; `models/v1_threshold_padrao/`: só y90 (DT, 4 KB).
- `reports/v2_…`: 3 CSVs (datados 2026-05-22) + 27 PNGs (matriz confusão, importância, curva threshold).
- `reports/v1_…`: 1 CSV (2026-05-22, com colunas `F1 (CV médio)`/`(CV desvio)`) + 6 PNGs.

---

## 9. Auditoria de vazamento e validade

| # | Verificação | Veredito | Evidência |
|---|---|---|---|
| 1 | Normalização/SPI com dados do teste | **SIM** | `SPI30_APRX`: climatologia mensal (`mean`/`std` de P30) calculada sobre toda a série, incl. 2020–2024 (`02_build_features.py`, ver 6.2). Demais features não são normalizadas. |
| 2 | Janelas móveis não-causais | **NÃO** (exceto SPI) | Todas as `rolling` são trailing. A exceção é a climatologia do SPI (item 1). `[FATO]` |
| 3 | Dias quase idênticos espalhados entre treino/validação inflando métricas | **SIM (parcial)** | Treino e teste são separados no tempo (bom). Porém os positivos são **blocos contíguos de 15 dias** fortemente autocorrelacionados, e as features diárias são suaves. Nos folds do CV isso agrava a degeneração (item 8.1). Efetivamente há 3 "eventos" independentes, não 45 amostras. `[INFERÊNCIA]` |
| 4 | Seleção de modelo/threshold usando o teste | **SIM (modelo), NÃO (threshold)** | Threshold só no treino (bom). Mas o "melhor modelo" salvo é escolhido pelo **F1 no teste** (linha 343). `[FATO]` |
| 5 | Shuffle em algum ponto | **NÃO** | Split cronológico e `TimeSeriesSplit`; nenhum `shuffle=True`. `[FATO]` |
| 6 | Nº de eventos de crise distintos no teste | **1** | Só o evento de 2021 (15 positivos/horizonte) cai no teste (início em 2020-07-31). Os de 2003-04 e 2014-15 ficam no treino. `[FATO]` |

> **Vazamento adicional não listado, porém grave:** a **mudança de escala por nº de estações**
> (precip/radiação somadas; RAD_SUM=0 no teste) não é "vazamento" clássico, mas é um **artefato de
> construção** que separa artificialmente treino e teste e invalida a comparação. `[INFERÊNCIA]`

---

## 10. Aplicação Streamlit

### 10.1 Como rodar e páginas `[FATO]`

`streamlit run app/Home.py`. Quatro páginas:
- **Home** — índice de risco do período: calcula `predict_proba` em todas as linhas do período
  escolhido, tira a média e normaliza pelo threshold (`min(prob_media/threshold, 1)`); gauge + faixas.
- **1_Previsao** — abas "Upload CSV" (valida header `timestamp`+22 features e prevê linha a linha) e
  "Inserção manual" (formulário de 22 campos para 1 dia).
- **2_Visualizacao** — séries temporais das features com crises destacadas e sobreposição de rótulos.
- **3_Sobre** — texto do projeto.

Entradas do usuário: horizonte (y30/60/90), intervalo de datas, upload de CSV ou valores manuais, seleção de features.

### 10.2 Login e credenciais `[FATO]`

**Não existe login.** Busca por `login|senha|password|secret|authenticat|credential` em `app/` não
retorna nada. Não há `secrets.toml` versionado (está no `.gitignore`), nem `st.text_input(type="password")`.
A afirmação do documento de "App Streamlit com login" é **DIVERGENTE**.

### 10.3 Como carrega modelo/threshold e se prevê sobre dado novo `[FATO]`

`app/utils.py`: `carregar_modelo(target)` faz `joblib.load` de `models/v2_threshold_otimizado/modelo_crise_hidrica_{target}.pkl`
(cacheado) e retorna `(modelo, threshold)`. O app **prevê de verdade** (`modelo.predict_proba`) —
tanto sobre o histórico (Home/Visualização) quanto sobre dados novos do usuário (Previsão: CSV ou
manual). Não é resultado pré-calculado. `[FATO]`

> **Alerta de UX decorrente do threshold 0,01:** como todos os thresholds são 0,01 e as probabilidades
> típicas são baixas, o índice normalizado (`prob/0,01`) satura em "Risco Alto" com facilidade — o
> app pode exibir alarme para períodos normais. `[INFERÊNCIA]`

### 10.4 Subida local `[FATO]`

- O servidor **sobe sem erro**: `streamlit run app/Home.py` (venv de auditoria) respondeu
  `HTTP 200` em `/` e `ok` em `/_stcore/health`; os 3 `.pkl` carregam sem exceção; log limpo.
- **Screenshots:** `[NÃO VERIFICADO visualmente]`. O Chrome controlado neste ambiente não conseguiu
  renderizar `http://localhost:8599` (até o endpoint de *health* retornou "error page" no navegador),
  embora o `curl` acessasse normalmente — provável bloqueio de `localhost` para a extensão/navegador
  nesta máquina. Não há screenshots em `_auditoria_tmp/screenshots/`. A funcionalidade foi validada
  via servidor (HTTP 200 + carga dos modelos + ausência de exceções), não via captura de tela.

---

## 11. Reuso e extensibilidade

| Mudança | Nota (1 difícil → 5 trivial) | Justificativa / arquivos que mudariam |
|---|---|---|
| Trocar fonte de dados (ERA5, CHIRPS, NASA POWER) | **2** | Nomes de coluna do INMET estão hardcoded em `01`/`02` (dicionários `num_candidates`/`aggs`). Trocar a fonte exige reescrever todo o parsing e a agregação. Daria para abstrair, mas hoje não há camada de adaptação. |
| Adicionar variáveis hidrológicas (volume, vazão) | **3** | Basta juntar ao parquet de features por data e incluir na lista `FEATURES`. O gargalo é **obter e alinhar** a série (não existe no projeto); o código aceitaria colunas novas quase sem mudança. |
| Trocar alvo binário por regressão (volume em t+30) | **2** | `03` (rotulagem binária por janela), `04` (classificadores, F1/AUC, threshold) e o app (gauge de probabilidade) assumem classificação. Exige novo alvo, novas métricas (RMSE/MAE) e nova UI. |
| Rodar com dataset sintético com crises injetadas | **4** | O pipeline é data-driven; basta um parquet diário com as colunas esperadas + um `eventos.csv`. `03` e `04` rodam direto. `01`/`02` poderiam ser pulados. Bom para testes controlados. |
| Adicionar modelos (LightGBM, RegLog, LSTM) | **3** (árvores) / **2** (LSTM) | Modelos sklearn-like entram no dict `criar_modelos` com pouco esforço. LSTM quebra o padrão (precisa de janelas 3D, scaler, early stopping) e não se encaixa no laço atual nem em `predict_proba` simples. |

---

## 12. Divergências entre documento e código

| Afirmação do documento | Veredito | Valor real / observação |
|---|---|---|
| INMET 2003–2024, XLSX horários de estações de SP | **CONFIRMADO** | 22 arquivos, 5.548.008 linhas horárias, 2003-01-01…2024-12-31. |
| 8.036 dias de observações contínuas | **CONFIRMADO** | 8.036 dias diários, contínuos, sem buracos. |
| 22 variáveis preditoras | **CONFIRMADO** | Lista `FEATURES` (`app/utils.py`) e `04` = exatamente 22. |
| 3 eventos de crise (2003-04, 2014-15, 2021) | **CONFIRMADO** | `eventos.csv`. |
| 45 positivos por horizonte; y90: 45 pos / 6.957 neg / 1.034 NaN | **CONFIRMADO** | Idêntico (e igual para y30 e y60). |
| Split 80/20: treino ~5.601 (2003–2020) / teste ~1.401 (2021–2024) | **PARCIAL / DIVERGENTE** | Tamanhos CONFIRMADOS (5.601/1.401). Mas o teste começa em **2020-07-31**, não em 2021 — inclui ~5 meses de 2020. |
| TimeSeriesSplit 5 folds; threshold 0,01–0,99 maximizando F1 médio | **CONFIRMADO (com ressalva)** | É exatamente isso; porém 4/5 folds têm 0 positivos → F1 médio 0 → threshold degenera para 0,01. |
| Modelos: DecisionTree, RandomForest (principal), XGBoost | **PARCIAL** | Os 3 existem. "Principal" só vale para y30 (RF é o salvo); em y60/y90 o salvo é **DecisionTree**. |
| Melhor modelo salvo com joblib + threshold | **CONFIRMADO** | `joblib.dump({"modelo":..., "threshold":...})`. |
| AUC-ROC 0,68–0,70 em 30 dias | **CONFIRMADO** | CSV y30: RF 0,682 / XGB 0,700. (Mas y60/y90 ~aleatório; F1≈0 em tudo.) |
| App Streamlit com login, criação de análise e resultados | **DIVERGENTE** | Há "análise" (previsão) e resultados, mas **não há login** nenhum. |
| "Estação São Paulo – Mirante (A701)" (README/app) | **DIVERGENTE / FALSO** | Os dados têm ~4–43 estações empilhadas sem identificador (seção 4.2). |

---

## 13. Dívida técnica e riscos

**ALTO**
- **Dados pesados versionados:** `.git` = **712 MB** (os XLSX de 439 MB e o parquet de 96 MB estão no
  histórico). Clone/backup caríssimos; difícil reverter. Deveriam sair via `.gitignore` + DVC/Git-LFS
  ou armazenamento externo.
- **Reprodutibilidade quebrada:** `.venv` versionado aponta para a máquina do autor
  (`C:\Users\Gustavo\anaconda3`, cp39) e não roda em outro lugar. Versionar o `.venv` é um erro.
- **Sem testes automatizados:** nenhum `test_*.py`, nenhum CI. Nada garante que uma mudança não
  quebre o pipeline.
- **Validade científica:** agregação soma-entre-estações, RAD_SUM=0 no teste, SPI com vazamento, CV
  degenerada e só 3 eventos — o resultado atual não sustenta conclusão. (Detalhado nas seções 4–9.)

**MÉDIO**
- **Backend matplotlib Tk hardcoded:** `04` quebra em headless sem `MPLBACKEND=Agg`.
- **Caminhos e hiperparâmetros hardcoded; sem arquivo de config** (seção 3.2).
- **`fillna(0)` global:** trata faltante como zero em features onde 0 tem significado (SPI, dry spells),
  mascarando problemas de dados.
- **Seleção do "melhor modelo" pelo F1 de teste** (seção 8.2).
- **Duas grafias de coluna de radiação** silenciosamente ignoradas (seção 4.4) — falta checagem de schema.

**BAIXO**
- **Código morto / duplicado:** `v1_threshold_padrao` (só y90) coexiste com v2; dicionário `_cfg` de
  faixas manuais em `1_Previsao.py`; `MODEL_PATH` legada em `utils.py`.
- **Documento × realidade:** README/app citam "A701/login" inexistentes (seções 10.2, 12).
- Não há notebooks fora do pipeline (nenhum notebook). Não há segredos versionados
  (`secrets.toml` está no `.gitignore`); nenhuma chave/token foi encontrada (nada foi lido).

---

## 14. Anexo: trechos de código centrais

### 14.1 Agregação entre estações — `02_build_features.py: hourly_to_daily`
```python
aggs = {p:'sum', t:'mean', tmaxh:'max', tminh:'min', urh:'mean',
        rad:'sum', press:'mean', vmed:'mean', vraj:'max'}
aggs = {k: v for k, v in aggs.items() if k in cols}
df = df.set_index('timestamp')
by_station = bool(station_col) and (station_col in cols)
if by_station:
    daily = df.groupby([station_col, pd.Grouper(freq='D')]).agg(aggs).reset_index()
else:
    # RAMO EXECUTADO: sem coluna de estação → agrega global por dia (mistura estações)
    daily = df.resample('D').agg(aggs).reset_index()
```

### 14.2 Features e SPI — `02_build_features.py: add_features_and_spi` (trechos)
```python
def roll_sum(s, w):               # janelas trailing (causais)
    return s.rolling(w, min_periods=int(w*0.7)).sum()
for w in [7, 15, 30, 90]:
    daily[f'P{w}'] = _apply_by(daily, grp, 'PRECIP_DIARIA', lambda s: roll_sum(s, w))

def dry_streak(s):
    is_dry = (s < 1.0).astype(float)
    c = (is_dry == 0).cumsum()
    return is_dry.groupby(c).cumcount() + is_dry
daily['DRY_STREAK_CUR'] = _apply_by(daily, grp, 'PRECIP_DIARIA', dry_streak)
daily['DRY30_MAX'] = _apply_by(daily, grp, 'DRY_STREAK_CUR', lambda s: roll_max(s,30,21))
daily['DRY90_MAX'] = _apply_by(daily, grp, 'DRY_STREAK_CUR', lambda s: roll_max(s,90,63))

# SPI30 — VAZAMENTO: climatologia sobre TODA a série (incl. teste)
daily['mes'] = daily['timestamp'].dt.month
clim = (daily.dropna(subset=['P30']).groupby(['mes'])['P30']
            .agg(['mean','std']).rename(columns={'mean':'P30_MEAN_MES','std':'P30_STD_MES'})
            .reset_index())
daily = daily.merge(clim, on=['mes'], how='left')
daily['SPI30_APRX'] = (daily['P30'] - daily['P30_MEAN_MES']) / daily['P30_STD_MES']
```

### 14.3 Rotulagem — `03_build_labels.py: label_pre_evento`
```python
for h in horizons:                                  # (30,60,90)
    win_ini = t0_ev - pd.Timedelta(days=h + w_pre)  # w_pre=7
    win_fim = t0_ev - pd.Timedelta(days=h - w_pos)  # w_pos=7  → janela de 15 dias
    mask_time = (out['timestamp'] >= win_ini) & (out['timestamp'] <= win_fim)
    out.loc[mask_sys & mask_time, f'y{h}'] = 1
if exclude_during and pd.notna(t1_ev):
    mask_dur = (out['timestamp'] >= t0_ev) & (out['timestamp'] <= t1_ev)
    out.loc[mask_sys & mask_dur, [f'y{h}' for h in horizons]] = np.nan
```

### 14.4 Carga, features e split — `04_train_model.py`
```python
df = df.dropna(subset=[alvo]).reset_index(drop=True)      # remove dias de crise (NaN)
colunas_excluir = ["timestamp","y30","y60","y90","mes","P30_MEAN_MES","P30_STD_MES"]
colunas_features = [c for c in df.columns
                    if c not in colunas_excluir and pd.api.types.is_numeric_dtype(df[c])]  # = 22
X = df[colunas_features].copy().fillna(0)                 # NaN → 0
y = df[alvo].astype(int)

def split_temporal(X, y, timestamps, pct_treino=0.8):     # 80/20 cronológico, sem shuffle
    n_treino = int(len(X) * pct_treino)
    return X.iloc[:n_treino], X.iloc[n_treino:], y.iloc[:n_treino], y.iloc[n_treino:], ...
```

### 14.5 Busca de threshold — `04_train_model.py: otimizar_threshold`
```python
tscv = TimeSeriesSplit(n_splits=5)
thresholds = np.arange(0.01, 1.00, 0.01)
f1_por_threshold = {t: [] for t in thresholds}
for fold, (idx_tr, idx_val) in enumerate(tscv.split(X_treino), 1):
    modelo_fold = type(modelo)(**modelo.get_params())
    modelo_fold.fit(X_treino.iloc[idx_tr], y_treino.iloc[idx_tr])
    y_prob_val = modelo_fold.predict_proba(X_treino.iloc[idx_val])[:, 1]
    for t in thresholds:
        f1_por_threshold[t].append(f1_score(y_treino.iloc[idx_val], (y_prob_val>=t).astype(int), zero_division=0))
f1_medio = {t: np.mean(f1_por_threshold[t]) for t in thresholds}
melhor_threshold = max(f1_medio, key=f1_medio.get)        # 4/5 folds têm 0 positivos → cai em 0,01
```

### 14.6 Avaliação e seleção — `04_train_model.py`
```python
def avaliar_modelo(modelo, X_teste, y_teste, nome, threshold=0.5):
    y_prob = modelo.predict_proba(X_teste)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    auc = roc_auc_score(y_teste, y_prob); f1 = f1_score(y_teste, y_pred, zero_division=0)
    cm = confusion_matrix(y_teste, y_pred)
    ...
melhor = max(resultados, key=lambda r: r["f1"])           # seleção pelo F1 DO TESTE
joblib.dump({"modelo": melhor["modelo_obj"], "threshold": melhor["threshold"]}, caminho_modelo)
```

### 14.7 Onde o app chama o modelo — `app/utils.py` + `app/Home.py`
```python
# utils.py
@st.cache_resource
def carregar_modelo(target="y90"):
    pkg = joblib.load(MODEL_PATHS.get(target, MODEL_PATH))
    return pkg["modelo"], float(pkg["threshold"])
def prever(modelo, threshold, df):
    X = df[FEATURES].fillna(0)
    probs = modelo.predict_proba(X)[:, 1]
    ...
# Home.py
modelo, threshold = carregar_modelo(horizonte)
X = df_periodo[FEATURES].fillna(0)
probs = modelo.predict_proba(X)[:, 1]
indice = min(prob_media / threshold, 1.0) if threshold > 0 else 0.0   # threshold=0,01 → satura fácil
```

---

### Apêndice — reprodução
Arquivos em `_auditoria_tmp/`: `inspect_raw.py`, `inspect_parquet.py`, `repro_train.py`
(+ `repro_train_out.txt`), `repro_pipeline/` (saídas dos passos 02–04 redirecionadas). O venv de
auditoria (`_auditoria_tmp/venv`, Python 3.13) foi criado apenas para esta auditoria e pode ser
descartado. Nenhum artefato do projeto foi alterado.
