#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_all.py — Orquestra o pipeline_v2 na ordem correta.

Por padrao roda o processamento e os experimentos de VALIDACAO (nao baixa dados nem abre
o teste). Use flags para incluir a coleta (--coleta) ou abrir o teste (--abrir-teste).
O teste de 2023+ so deve ser aberto uma vez, depois do congelamento (rigor, CLAUDE.md).

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


def roda(etapas):
    for rotulo, cmd in etapas:
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
    roda(etapas)
    print("\nOK: pipeline_v2 concluido.")


if __name__ == "__main__":
    main()
