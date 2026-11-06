#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_all.py — Orquestra o pipeline_v2 na ordem correta.

Por padrao roda o processamento e os experimentos de VALIDACAO (nao baixa dados nem abre
o teste). Use flags para incluir a coleta (--coleta) ou abrir o teste (--abrir-teste).
O teste de 2023+ so deve ser aberto uma vez, depois do congelamento (rigor, CLAUDE.md).
Antes de cada etapa confere os arquivos de entrada; se faltar algum, para com mensagem
dizendo qual arquivo falta e de onde ele vem (dados brutos e da v1 sao conferidos ja no inicio).

Exemplos:
  python pipeline_v2/run_all.py                 # dataset + Exp 0/1/2/3/4 + LSTM (validacao)
  python pipeline_v2/run_all.py --coleta        # inclui 00_download_dados --all (rede)
  python pipeline_v2/run_all.py --congelar      # inclui 03 (grava congelamento.json)
  python pipeline_v2/run_all.py --abrir-teste   # abre o teste (decisao deliberada)
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

AQUI = Path(__file__).resolve().parent
ROOT = AQUI.parent

# (rotulo, comando) — comando relativo ao diretorio pipeline_v2
ETAPAS_COLETA = [("coleta SAR+ERA5+POWER", ["00_download_dados.py", "--all"])]
ETAPAS_BASE = [
    ("dataset diario", ["01_build_dataset_v2.py"]),
    ("Exp 0 (diagnostico v1)", ["exp0_diagnostico_v1.py"]),
    ("Exp 2 (validacao)", ["02_treino_v2.py"]),
    ("Exp 3 (residual)", ["02b_analise_exp3.py"]),
    ("Exp 3 (antecedencia pareada)", ["02d_antecedencia_pareada.py"]),
    ("LSTM", ["02c_lstm.py"]),
    ("Exp 1 (sintetico)", ["04_experimento_sintetico.py"]),
    ("Exp 1 (ajuste phi_ef + gemeo)", ["04b_sintetico_ajuste.py"]),
    ("Exp 4 (outros reservatorios)", ["exp4_outros_reservatorios.py"]),
]
ETAPAS_CONGELAR = [("congelamento", ["03_congelar_e_testar.py"])]
ETAPAS_TESTE = [("abrir teste", ["03_congelar_e_testar.py", "--abrir-teste"]),
                ("diagnostico pos-hoc do teste", ["03b_teste_diagnostico.py"])]

# Origem de cada entrada, para a mensagem de erro dizer de onde o arquivo vem.
ORIGEM_COLETA = ("baixado pelo passo 1 do README (python pipeline_v2/00_download_dados.py --all); "
                 "no pacote do projeto ja vem em data/raw_v2/")
ORIGEM_V1 = ("dados da v1 (INMET/BDMEP, 2003-2024), fora do git; no pacote do projeto ja vem em "
             "legado_v1/data/. Fonte e como regenerar: legado_v1/README.md")
ORIGEM_V1_REPORTS = "relatorios da v1, versionados no git em legado_v1/reports/threshold_otimizado/"
ORIGEM_DATASET = "gerado pelo passo 2 do README (python pipeline_v2/01_build_dataset_v2.py)"
ORIGEM_SINTETICO = "gerado pelo Exp 1 (python pipeline_v2/04_experimento_sintetico.py)"
ORIGEM_CONGELAMENTO = ("gerado por python pipeline_v2/03_congelar_e_testar.py (sem --abrir-teste); "
                       "versionado no git em reports_v2/final/")

_DATASET = [("data/processed_v2/dataset_diario.parquet", ORIGEM_DATASET),
            ("data/processed_v2/episodios_crise.csv", ORIGEM_DATASET),
            ("data/processed_v2/episodios_crise40.csv", ORIGEM_DATASET)]

# Entradas de cada script (caminho relativo a raiz, origem). Aceita glob.
ENTRADAS = {
    "00_download_dados.py": [],
    "01_build_dataset_v2.py": [("data/raw_v2/sar_cantareira_sistema_diario.csv", ORIGEM_COLETA),
                               ("data/raw_v2/openmeteo_pontos_diario.csv", ORIGEM_COLETA),
                               ("data/raw_v2/nasapower_pontos_diario.csv", ORIGEM_COLETA)],
    "exp0_diagnostico_v1.py": [
        ("legado_v1/data/features/inmet_sp_daily.parquet", ORIGEM_V1),
        ("legado_v1/data/features/inmet_sp_daily_features.parquet", ORIGEM_V1),
        ("legado_v1/data/features/inmet_sp_daily_labels.parquet", ORIGEM_V1),
        ("legado_v1/data/interim/inmet_sp_hourly_clean.parquet", ORIGEM_V1),
        *[(f"legado_v1/reports/threshold_otimizado/comparativo_modelos_y{h}_*.csv",
           ORIGEM_V1_REPORTS) for h in (30, 60, 90)]],
    "02_treino_v2.py": _DATASET,
    "02b_analise_exp3.py": _DATASET,
    "02d_antecedencia_pareada.py": _DATASET,
    "02c_lstm.py": _DATASET,
    "04_experimento_sintetico.py": _DATASET[:1],
    "04b_sintetico_ajuste.py": [*_DATASET[:1],
                                ("reports_v2/sintetico/grade_sintetico.csv", ORIGEM_SINTETICO)],
    "exp4_outros_reservatorios.py": [("data/raw_v2/sar_cantareira_diario.csv", ORIGEM_COLETA),
                                     ("data/raw_v2/openmeteo_pontos_diario.csv", ORIGEM_COLETA)],
    "03_congelar_e_testar.py": _DATASET,
    "03b_teste_diagnostico.py": [*_DATASET,
                                 ("reports_v2/final/congelamento.json", ORIGEM_CONGELAMENTO)],
}


def entradas_faltando(script, raiz=ROOT):
    """Lista (caminho, origem) das entradas do script que nao existem sob a raiz."""
    faltam = []
    for caminho, origem in ENTRADAS[script]:
        existe = (any(raiz.glob(caminho)) if any(c in caminho for c in "*?")
                  else (raiz / caminho).is_file())
        if not existe:
            faltam.append((caminho, origem))
    return faltam


def _interrompe(cabecalho, faltam):
    print(f"\n!!! {cabecalho}: falta(m) {len(faltam)} arquivo(s) de entrada:", flush=True)
    for caminho, origem in faltam:
        print(f"    - {caminho}\n      origem: {origem}", flush=True)
    print("!!! interrompendo antes de rodar a etapa.", flush=True)
    sys.exit(2)


def confere_entradas(rotulo, script, raiz=ROOT):
    """Interrompe com mensagem clara se faltar alguma entrada do script."""
    faltam = entradas_faltando(script, raiz)
    if faltam:
        _interrompe(f"etapa '{rotulo}' ({script}) nao pode rodar", faltam)


def pre_checagem(etapas, coleta, raiz=ROOT):
    """Confere ja no inicio as entradas que nenhuma etapa do run gera (dados brutos e v1),
    para nao interromper so depois de etapas longas."""
    externas = {ORIGEM_V1, ORIGEM_V1_REPORTS} | (set() if coleta else {ORIGEM_COLETA})
    faltam = []
    for _, cmd in etapas:
        faltam += [f for f in entradas_faltando(cmd[0], raiz) if f[1] in externas and f not in faltam]
    if faltam:
        _interrompe("pre-checagem", faltam)


def roda(etapas):
    for rotulo, cmd in etapas:
        confere_entradas(rotulo, cmd[0])
        print(f"\n{'='*70}\n>>> {rotulo}: {' '.join(cmd)}\n{'='*70}", flush=True)
        t = time.time()
        r = subprocess.run([sys.executable, str(AQUI / cmd[0]), *cmd[1:]])
        print(f"<<< {rotulo} terminou em {time.time()-t:.0f}s (codigo {r.returncode})", flush=True)
        if r.returncode != 0:
            print(f"!!! etapa '{rotulo}' falhou; interrompendo.", flush=True)
            sys.exit(r.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coleta", action="store_true", help="Inclui a coleta de dados (rede).")
    ap.add_argument("--congelar", action="store_true", help="Inclui o congelamento (03).")
    ap.add_argument("--abrir-teste", action="store_true", help="Abre o teste 2023+ (deliberado).")
    args = ap.parse_args()

    etapas = []
    if args.coleta:
        etapas += ETAPAS_COLETA
    etapas += ETAPAS_BASE
    if args.congelar or args.abrir_teste:
        etapas += ETAPAS_CONGELAR
    if args.abrir_teste:
        etapas += ETAPAS_TESTE
    pre_checagem(etapas, args.coleta)
    roda(etapas)
    print("\nOK: pipeline_v2 concluido.")


if __name__ == "__main__":
    main()
