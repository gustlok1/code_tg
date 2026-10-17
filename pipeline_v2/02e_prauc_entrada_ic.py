#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02e_prauc_entrada_ic.py — Gera reports_v2/resultados/prauc_entrada_ic.csv: a PR-AUC
AGREGADA (out-of-fold) da ENTRADA em crise (limiar 40) na validacao, para B3 e XGBRes,
com IC95% por bootstrap em blocos de 90 dias. Esse arquivo era usado pelo texto (Exp 2/3,
Tabelas 9/11) mas nenhum script o regenerava; aqui a geracao fica versionada e reproduzivel.

Reproducao bit a bit do arquivo historico: um unico np.random.default_rng(seed) compartilhado,
na ordem h30 -> h60 -> h90 e modelos B3 -> XGBRes (parametros no config, bloco prauc_entrada_ic).

SO VALIDACAO. Nao abre nem toca o teste (2023+). Nao roda o 03.

Uso:
  python pipeline_v2/02e_prauc_entrada_ic.py             # compara com o arquivo atual (nao escreve)
  python pipeline_v2/02e_prauc_entrada_ic.py --escrever  # regrava o CSV
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import analise_exp3 as ax

ROOT = Path(__file__).resolve().parents[1]
CFGP = Path(__file__).resolve().parent / "config.yaml"
OUT = ROOT / "reports_v2" / "resultados" / "prauc_entrada_ic.csv"
COLS = ["horizonte", "modelo", "prevalencia", "pr_auc", "ic95_lo", "ic95_hi", "lo_acima_prev"]


def _utf8():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass


def calcular(df, cfg):
    pb = cfg.get("prauc_entrada_ic", {})
    seed = int(pb.get("seed", 0)); n_boot = int(pb.get("n_boot", 2000))
    bloco = int(pb.get("bloco", 90)); limiar = int(pb.get("limiar", 40))
    rng = np.random.default_rng(seed)   # UNICO rng, compartilhado entre horizontes e modelos
    linhas = []
    for h in cfg["horizontes"]:
        _, oof = ax.validar_residual(df, cfg, h, src="era5")
        linhas += ax.prauc_entrada_ic_horizonte(oof, h, rng, limiar=limiar,
                                                 n_boot=n_boot, bloco=bloco)
    return pd.DataFrame(linhas)[COLS]


def main():
    _utf8()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--escrever", action="store_true", help="regrava o CSV (padrao: so compara)")
    args = ap.parse_args()
    cfg = yaml.safe_load(CFGP.read_text(encoding="utf-8"))
    df = pd.read_parquet(ROOT / cfg["caminhos"]["processed_v2_dataset"]); df["data"] = pd.to_datetime(df["data"])
    novo = calcular(df, cfg)
    print("\n[prauc_entrada_ic recomputado]\n" + novo.to_string(index=False))
    if OUT.exists():
        atual = pd.read_csv(OUT)[COLS].round(4).reset_index(drop=True)
        igual = atual.equals(novo[COLS].round(4).reset_index(drop=True))
        print(f"\nComparacao com {OUT.name}: {'IDENTICO' if igual else 'DIFERENTE'}")
        if not igual:
            print("ATENCAO: diferente do arquivo atual. NAO sobrescreva sem decisao do Victor.")
    if args.escrever:
        novo.to_csv(OUT, index=False, encoding="utf-8")
        print(f"\nGravado: {OUT}")


if __name__ == "__main__":
    main()
