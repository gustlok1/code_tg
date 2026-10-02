# pipeline_v2 — Previsão de crises hídricas no Sistema Cantareira (reconstrução)

Reconstrução do TG2 após o diagnóstico da v1 (`RELATORIO_ESTADO_ATUAL.md`). Alvo objetivo (volume
útil do SAR/ANA), validação com purga, baselines honestos e teste intocável até o congelamento.
Decisões em `DECISOES.md`; regras do projeto em `CLAUDE.md`.

## Como rodar
Requer Python 3.11+ e um ambiente a partir do `requirements.txt` (o `.venv` versionado aponta para
outra máquina e não roda; crie o seu).

```
python -m venv .venv_local
.venv_local\Scripts\activate            # Windows
pip install -r requirements.txt
# torch CPU:
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu

python pipeline_v2/run_all.py --coleta   # coleta + dataset + experimentos (validacao)
```

`run_all.py` sem `--coleta` assume os dados já baixados. O teste (2023+) só abre com
`python pipeline_v2/03_congelar_e_testar.py --abrir-teste`, depois do congelamento.

## Estrutura
- `config.yaml`: parâmetros (horizontes, limiares, folds, split, pontos da bacia, caminhos).
- Módulos importáveis (lógica testada): `dataset_v2.py`, `modelagem_v2.py`, `analise_exp3.py`,
  `lstm_v2.py`, `sintetico_v2.py`, `transformadores.py`.
- Orquestradores numerados:
  - `00_download_dados.py`: SAR (1984+), ERA5-Land (Open-Meteo), NASA POWER; `_manifest.json`.
  - `01_build_dataset_v2.py`: calendário diário, features causais, alvos e episódios automáticos.
  - `02_treino_v2.py` (Exp 2): validação janela crescente + purga; B1/B2/B3 e modelos.
  - `02b_analise_exp3.py` (Exp 3): modelo residual sobre o B3, skill condicional, robustez POWER.
  - `02c_lstm.py` + `02d_antecedencia_pareada.py`: LSTM e teste pareado da antecedência.
  - `03_congelar_e_testar.py`: congela escolhas e abre o teste uma única vez; `03b_teste_diagnostico.py`.
  - `04_experimento_sintetico.py` + `04b_sintetico_ajuste.py` (Exp 1): sintético, φ_ef e gêmeo.
  - `exp0_diagnostico_v1.py` (Exp 0): figuras do diagnóstico da v1.
  - `exp4_outros_reservatorios.py` (Exp 4): o mesmo pipeline em outros reservatórios.
  - `run_all.py`: roda tudo na ordem.
- `tests/`: pytest (26 testes). Rode `python -m pytest pipeline_v2/tests -q`.

## Dados (fora do git)
`data/raw_v2/`, `data/processed_v2/` e `data/sintetico/` não são versionados (ver `.gitignore`).
São reproduzíveis por `run_all.py --coleta`. Figuras, tabelas e resumos finais em `reports/v2/`.

## Rigor
Teste aberto uma única vez pelo `03`, depois do congelamento; nada muda depois. Nenhuma estatística de
série inteira fora de `transformadores.py` (fit no treino). Baselines sempre reportados; skill contra
o B3 é a régua. Dado sintético sempre rotulado "DADOS SINTÉTICOS". Ver `reports/v2/final/ficha_tecnica.md`.
