#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verificar_reproducao.py — Reproducibilidade (item 12). Recomputa, no ambiente atual, as
escolhas do congelamento (configs e thresholds) e as metricas do teste, e compara com os
valores JA COMMITADOS. NAO sobrescreve o modelo congelado nem reabre nada de forma a muda-lo.
Saida: reports_v2/final/reproducao.md.
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, mean_absolute_error

import analise_exp3 as ax

ROOT = Path(__file__).resolve().parents[1]
CFGP = Path(__file__).resolve().parent / "config.yaml"
FINAL = ROOT / "reports_v2" / "final"


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = yaml.safe_load(CFGP.read_text(encoding="utf-8"))
    df = pd.read_parquet(ROOT / cfg["caminhos"]["processed_v2_dataset"]); df["data"] = pd.to_datetime(df["data"])
    cong = json.loads((FINAL / "congelamento.json").read_text(encoding="utf-8"))
    met_ref = pd.read_csv(FINAL / "metricas_teste.csv")

    import pandas as _pd
    linhas = []
    tudo_ok = True
    for h in cfg["horizontes"]:
        # 1) config escolhida reproduz?
        esc = ax.melhor_config_residual(df, cfg, h)["XGBRes"]["config"]
        ref_cfg = cong["modelos"]["XGBRes"][str(h)]["config"]
        ok_cfg = esc == ref_cfg
        tudo_ok &= ok_cfg
        linhas.append({"item": f"config XGBRes h{h}", "comprometido": json.dumps(ref_cfg),
                       "reproduzido": json.dumps(esc), "igual": ok_cfg})
        # 2) thresholds reproduzem?
        _, oof = ax.validar_residual(df, cfg, h, "era5")
        for limiar in (40, 30):
            crise = _pd.Series(oof["real"].values < limiar, index=pd.to_datetime(oof["data"]).values)
            for modelo in ("B3", "XGBRes"):
                thr = round(float(ax._threshold_vol(oof, modelo, crise)), 2)
                ref = cong["thresholds_volume_alerta"][str(limiar)][str(h)][modelo]
                ok = abs(thr - ref) < 1e-9
                tudo_ok &= ok
                linhas.append({"item": f"thr {modelo} lim{limiar} h{h}", "comprometido": ref,
                               "reproduzido": thr, "igual": ok})
        # 3) metricas do teste reproduzem?
        ooft = ax.testar_residual(df, cfg, h, ref_cfg)
        real = ooft["real"].values
        mae_b3 = mean_absolute_error(real, ooft["B3"].values)
        for modelo in ("B3", "XGBRes"):
            mae = mean_absolute_error(real, ooft[modelo].values)
            skill = round(1 - mae / mae_b3, 4) if mae_b3 > 0 else np.nan
            y = (real < 40).astype(int); ent = ooft["vol_t"].values > 40
            prauc = (round(float(average_precision_score(y[ent], -ooft[modelo].values[ent])), 4)
                     if ent.sum() and 0 < y[ent].sum() < ent.sum() else np.nan)
            ref_row = met_ref[(met_ref.horizonte == h) & (met_ref.modelo == modelo)].iloc[0]
            for nome, val, refv in [("MAE", round(float(mae), 3), float(ref_row["MAE"])),
                                    ("skill_B3", skill, float(ref_row["skill_B3"])),
                                    ("PR_AUC_entrada40", prauc, float(ref_row["PR_AUC_entrada40"]))]:
                ok = abs(val - refv) < 5e-3
                tudo_ok &= ok
                linhas.append({"item": f"teste {nome} {modelo} h{h}", "comprometido": refv,
                               "reproduzido": val, "igual": ok})

    tab = pd.DataFrame(linhas)
    difs = tab[~tab["igual"]]
    md = ["# Reproducao do congelamento e do teste (item 12)", "",
          f"Ambiente: pandas {pd.__version__}. Comparacao entre os valores commitados e os recomputados.",
          f"Resultado geral: {'TODOS IGUAIS' if tudo_ok else 'HA DIFERENCAS'} "
          f"({len(tab) - len(difs)}/{len(tab)} itens identicos).", "",
          "| item | comprometido | reproduzido | igual |", "|---|---|---|---|"]
    for _, r in tab.iterrows():
        md.append(f"| {r['item']} | {r['comprometido']} | {r['reproduzido']} | {'sim' if r['igual'] else 'NAO'} |")
    if len(difs):
        md += ["", "## Diferencas", ""] + [f"- {r['item']}: commit={r['comprometido']} vs repro={r['reproduzido']}"
                                           for _, r in difs.iterrows()]
    (FINAL / "reproducao.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[:4]))
    print(f"\nDiferencas: {len(difs)} de {len(tab)} itens. Relatorio: {FINAL/'reproducao.md'}")


if __name__ == "__main__":
    main()
