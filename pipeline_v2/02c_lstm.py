#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02c_lstm.py — LSTM residual no mesmo arcabouço (folds+purga+transformadores no treino),
comparado a XGBRes e B3 (geral e condicional), com antecedência pareada vs B3.
Orçamento de tempo: se o LSTM não superar o XGBRes na validação, registra e segue.
NÃO abre o teste 2023+.

Uso:  python pipeline_v2/02c_lstm.py [--budget 1200]
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analise_exp3 as ax
import lstm_v2 as lv

ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path(__file__).resolve().parent / "config.yaml"


def fig_skill_lstm(cond, horizontes, out):
    paineis = [("geral", "Geral"), ("seca", "Seca (SPI-12 < −1)"),
               ("pre_episodio", "Pré-episódio")]
    cores = {"XGBRes": "#2ca02c", "LSTM": "#9467bd"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax_, (sub, tit) in zip(axes, paineis):
        for m in ("XGBRes", "LSTM"):
            d = cond[(cond.subconjunto == sub) & (cond.modelo == m)].sort_values("horizonte")
            if not len(d):
                continue
            yerr = np.vstack([d.skill_B3 - d.ic95_lo, d.ic95_hi - d.skill_B3])
            ax_.errorbar(d.horizonte, d.skill_B3, yerr=yerr, marker="o", capsize=4,
                         color=cores[m], label=m)
        ax_.axhline(0, color="#d62728", ls="--", lw=1, label="B3")
        ax_.set_title(tit); ax_.set_xlabel("Horizonte"); ax_.set_xticks(horizontes); ax_.grid(alpha=.3)
    axes[0].set_ylabel("Skill vs B3 (IC95%)"); axes[0].legend(fontsize=8)
    fig.suptitle("LSTM vs XGBRes vs B3 — skill por subconjunto")
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=1200, help="Orçamento de tempo do LSTM (s).")
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--stride", type=int, default=5)
    args = ap.parse_args()
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
    epi40 = pd.read_csv(proc / "episodios_crise40.csv")
    epi30 = pd.read_csv(proc / "episodios_crise.csv")
    horizontes = cfg["horizontes"]

    logging.info(f"Treinando LSTM (budget={args.budget}s, hidden={args.hidden}, stride={args.stride})...")
    linhas_l, oof_lstm, info = lv.validar_lstm(df, cfg, hidden=args.hidden, stride_treino=args.stride,
                                               budget_s=args.budget, log=logging.info)
    pd.DataFrame(linhas_l).to_csv(res / "lstm_metricas.csv", index=False, encoding="utf-8")
    logging.info(f"LSTM: completo={info['completo']} tempo={info['tempo_s']}s")

    # XGBRes/B3 (recomputa OOF do Exp 3) e comparação condicional
    cond_rows, pareado_rows = [], []
    for h in horizontes:
        _, oof_x = ax.validar_residual(df, cfg, h, src="era5")
        if h in oof_lstm:
            m = oof_x.merge(oof_lstm[h][["data", "LSTM"]], on="data", how="inner")
            cond_rows += ax.skill_condicional(m, epi40, ["XGBRes", "LSTM"], h, seed=h)
            for limiar in (40, 30):
                _, rp = ax.antecedencia_pareada(m, base, epi40 if limiar == 40 else epi30,
                                                h, limiar, modelo_a="LSTM", modelo_b="B3")
                pareado_rows.append(rp)

    cond = pd.DataFrame(cond_rows)
    cond.to_csv(res / "skill_condicional_lstm.csv", index=False, encoding="utf-8")
    pd.DataFrame(pareado_rows).to_csv(res / "antecedencia_pareada_lstm.csv", index=False, encoding="utf-8")

    # tabela LSTM vs XGBRes vs B3 (geral, skill_B3 médio dos folds)
    lx = pd.DataFrame(linhas_l)
    resumo_lstm = (lx.groupby("horizonte", as_index=False)["skill_B3"].mean().round(4)
                   .rename(columns={"skill_B3": "LSTM"})) if len(lx) else pd.DataFrame()
    # XGBRes geral vem do skill_condicional (subconjunto geral)
    xgb_geral = (cond[(cond.subconjunto == "geral") & (cond.modelo == "XGBRes")]
                 [["horizonte", "skill_B3"]].rename(columns={"skill_B3": "XGBRes"}))
    lstm_geral = (cond[(cond.subconjunto == "geral") & (cond.modelo == "LSTM")]
                  [["horizonte", "skill_B3"]].rename(columns={"skill_B3": "LSTM"}))
    comp = xgb_geral.merge(lstm_geral, on="horizonte", how="outer")
    comp["B3"] = 0.0
    comp.to_csv(res / "lstm_vs_xgb_resumo.csv", index=False, encoding="utf-8")

    # veredito (budget): o LSTM superou o XGBRes na validação?
    lstm_med = comp["LSTM"].mean() if "LSTM" in comp else np.nan
    xgb_med = comp["XGBRes"].mean() if "XGBRes" in comp else np.nan
    superou = bool(lstm_med > xgb_med) if (lstm_med == lstm_med and xgb_med == xgb_med) else False
    verdict = (f"LSTM {'SUPEROU' if superou else 'NÃO superou'} o XGBRes na validação "
               f"(skill_B3 geral médio: LSTM={lstm_med:.4f} vs XGBRes={xgb_med:.4f}). "
               f"completo={info['completo']} tempo={info['tempo_s']}s "
               f"(hidden={info['hidden']}, seeds={info['seeds']}, stride={info['stride_treino']}).")
    (res / "lstm_veredito.txt").write_text(verdict + "\n", encoding="utf-8")

    # figura
    if len(cond):
        fig_skill_lstm(cond, horizontes, res / "skill_lstm.png")

    print("\n" + "=" * 78)
    print("RESUMO LSTM — mesmo arcabouço (validação; teste 2023+ INTOCADO)")
    print("=" * 78)
    print(verdict)
    print("\n[Geral] skill_B3 médio (LSTM vs XGBRes vs B3=0):")
    print(comp.round(4).to_string(index=False))
    print("\n[Condicional] skill_B3 (IC95%) LSTM:")
    for sub in ("geral", "seca", "pre_episodio"):
        s = cond[(cond.subconjunto == sub) & (cond.modelo == "LSTM")]
        if len(s):
            print(f"  {sub}: " + " | ".join(
                f"h{int(r.horizonte)}={r.skill_B3} [{r.ic95_lo},{r.ic95_hi}]" for _, r in s.iterrows()))
    print("\n[Antecedência pareada LSTM − B3]:")
    pr = pd.DataFrame(pareado_rows)
    if len(pr):
        print(pr[["limiar", "horizonte", "n_episodios", "mediana_dif", "ic95_lo", "ic95_hi",
                  "wilcoxon_p", "falso_alarme_abs_LSTM", "falso_alarme_abs_B3"]].to_string(index=False))
    print(f"\nCSVs/figuras em: {res}")
    print("=" * 78)


if __name__ == "__main__":
    main()
