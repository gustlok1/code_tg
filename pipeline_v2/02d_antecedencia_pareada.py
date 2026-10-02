#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02d_antecedencia_pareada.py — Teste PAREADO da antecedência (XGBRes − B3) por episódio,
limiares 40 e 30, h = 30/60/90: mediana da diferença, IC95% (bootstrap sobre episódios) e
Wilcoxon pareado; mais a taxa e o número absoluto de falso-alarme na validação.
Não abre o teste 2023+.

Saída: reports_v2/resultados/antecedencia_pareada.csv (+ _por_episodio.csv)
"""
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

import analise_exp3 as ax

ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path(__file__).resolve().parent / "config.yaml"


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S")
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    res = ROOT / cfg["caminhos"]["resultados_v2"]
    proc = ROOT / "data" / "processed_v2"
    df = pd.read_parquet(ROOT / cfg["caminhos"]["processed_v2_dataset"]); df["data"] = pd.to_datetime(df["data"])
    base = df[["data", "vol_util_pct_sistema"]].copy()
    epis = {30: pd.read_csv(proc / "episodios_crise.csv"),
            40: pd.read_csv(proc / "episodios_crise40.csv")}

    resumos, por_ep = [], []
    for h in cfg["horizontes"]:
        logging.info(f"OOF h={h}...")
        _, oof = ax.validar_residual(df, cfg, h, src="era5")
        for limiar in (40, 30):
            pe, rs = ax.antecedencia_pareada(oof, base, epis[limiar], h, limiar,
                                             modelo_a="XGBRes", modelo_b="B3")
            resumos.append(rs); por_ep += pe

    pd.DataFrame(resumos).to_csv(res / "antecedencia_pareada.csv", index=False, encoding="utf-8")
    pd.DataFrame(por_ep).to_csv(res / "antecedencia_pareada_por_episodio.csv", index=False,
                                encoding="utf-8")

    r = pd.DataFrame(resumos)
    print("\n" + "=" * 78)
    print("ANTECEDÊNCIA PAREADA (XGBRes − B3) — mediana da diferença, IC95%, Wilcoxon")
    print("=" * 78)
    print(r[["limiar", "horizonte", "n_episodios", "mediana_dif", "ic95_lo", "ic95_hi",
             "wilcoxon_p", "falso_alarme_ano_XGBRes", "falso_alarme_abs_XGBRes",
             "falso_alarme_ano_B3", "falso_alarme_abs_B3"]].to_string(index=False))
    print(f"\nSalvo em: {res}")
    print("=" * 78)


if __name__ == "__main__":
    main()
