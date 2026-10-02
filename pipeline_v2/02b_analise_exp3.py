#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02b_analise_exp3.py — Análise do Exp 3 (modelo RESIDUAL sobre o B3), avaliação
condicional (seca / pré-episódio) com IC95%, antecedência com B3 ao lado e robustez
POWER. NÃO congela modelo nem abre o teste 2023+ (continua intocado).

Uso:  python pipeline_v2/02b_analise_exp3.py
Lógica em analise_exp3.py (importável e testada).
"""
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

ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path(__file__).resolve().parent / "config.yaml"
MODELOS = ["RidgeRes", "XGBRes"]


def _utf8():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass


def fig_skill_condicional(cond, horizontes, melhor, out):
    paineis = [("geral", "Geral"), ("seca", "Seca (SPI-12 < −1)"),
               ("pre_episodio", "Pré-episódio (−180..+30 d)")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    cores = {"RidgeRes": "#1f77b4", "XGBRes": "#2ca02c"}
    for ax_, (sub, titulo) in zip(axes, paineis):
        for m in MODELOS:
            d = cond[(cond.subconjunto == sub) & (cond.modelo == m)].sort_values("horizonte")
            if not len(d):
                continue
            yerr = np.vstack([d["skill_B3"] - d["ic95_lo"], d["ic95_hi"] - d["skill_B3"]])
            ax_.errorbar(d["horizonte"], d["skill_B3"], yerr=yerr, marker="o",
                         capsize=4, color=cores[m], label=m,
                         lw=(2 if m == melhor else 1.2))
        ax_.axhline(0, color="#d62728", ls="--", lw=1, label="B3")
        ax_.set_title(titulo); ax_.set_xlabel("Horizonte (dias)"); ax_.set_xticks(horizontes)
        ax_.grid(alpha=0.3)
    axes[0].set_ylabel("Skill vs B3 (IC95% bootstrap em blocos de 90 d)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Exp 3 — skill do modelo residual vs B3 por subconjunto")
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


def fig_backtest(oof90, base, melhor, out):
    d = oof90.copy(); d["alvo"] = pd.to_datetime(d["data"]) + pd.Timedelta(days=90)
    ini, fim = pd.Timestamp("2013-01-01"), pd.Timestamp("2016-12-31")
    d = d[(d["alvo"] >= ini) & (d["alvo"] <= fim)].sort_values("alvo")
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(d["alvo"], d["real"], color="#1f3b73", lw=1.6, label="real (vol t+90)")
    ax.plot(d["alvo"], d["vol_t"], color="#7f7f7f", lw=1.0, ls=":", label="B1 (persistência)")
    ax.plot(d["alvo"], d["B3"], color="#ff7f0e", lw=1.1, ls="--", label="B3")
    ax.plot(d["alvo"], d[melhor], color="#d1495b", lw=1.2, label=f"{melhor} (residual+B3)")
    ax.axhline(30, color="#d62728", lw=0.7, ls=":")
    ax.set_title("Backtest 2013–16 (validação) — h=90: real, B1, B3 e melhor modelo")
    ax.set_ylabel("Volume útil (%)"); ax.set_xlabel("Data alvo (t+90)"); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


def main():
    _utf8()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    res = ROOT / cfg["caminhos"]["resultados_v2"]; res.mkdir(parents=True, exist_ok=True)
    proc = ROOT / "data" / "processed_v2"
    df = pd.read_parquet(ROOT / cfg["caminhos"]["processed_v2_dataset"])
    df["data"] = pd.to_datetime(df["data"])
    horizontes = cfg["horizontes"]
    base = df[["data", "vol_util_pct_sistema"]].copy()
    epi30 = pd.read_csv(proc / "episodios_crise.csv")
    epi40 = pd.read_csv(proc / "episodios_crise40.csv")

    logging.info("Validação do modelo residual (ERA5)...")
    met_rows, oof_por_h = [], {}
    for h in horizontes:
        linhas, oof = ax.validar_residual(df, cfg, h, src="era5")
        met_rows += linhas; oof_por_h[h] = oof

    met = pd.DataFrame(met_rows)
    met.to_csv(res / "residual_metricas.csv", index=False, encoding="utf-8")
    resumo = (met.groupby(["modelo", "horizonte"], as_index=False)[["MAE", "skill_B1", "skill_B3"]]
                 .mean().round(3))
    resumo.to_csv(res / "residual_vs_b3.csv", index=False, encoding="utf-8")
    melhor = (resumo[resumo.modelo.isin(MODELOS)].groupby("modelo")["skill_B3"].mean()
                     .idxmax())
    logging.info(f"Melhor modelo residual (skill_B3 médio): {melhor}")

    # avaliação condicional
    logging.info("Avaliação condicional (seca / pré-episódio) com bootstrap...")
    cond_rows = []
    for h in horizontes:
        cond_rows += ax.skill_condicional(oof_por_h[h], epi40, MODELOS, h, seed=h)
    cond = pd.DataFrame(cond_rows)
    cond.to_csv(res / "skill_condicional.csv", index=False, encoding="utf-8")

    # antecedência lado a lado (B3 + melhor), limiares 30 e 40
    logging.info("Antecedência por episódio (B3 ao lado)...")
    ant_rows = []
    for h in horizontes:
        ant_rows += ax.antecedencia_lado_a_lado(oof_por_h[h], base, epi30, ["B3", melhor], h, 30)
        ant_rows += ax.antecedencia_lado_a_lado(oof_por_h[h], base, epi40, ["B3", melhor], h, 40)
    ant = pd.DataFrame(ant_rows)
    ant.to_csv(res / "antecedencia_exp3.csv", index=False, encoding="utf-8")

    # robustez POWER (com e sem 1999), no melhor modelo
    logging.info("Robustez POWER (chuva POWER no lugar da ERA5)...")
    rob_rows = []
    for fonte, excluir in [("era5", ()), ("power", ()), ("power_sem1999", (1999,))]:
        src = "era5" if fonte == "era5" else "power"
        for h in horizontes:
            linhas, _ = ax.validar_residual(df, cfg, h, src=src, excluir_anos=excluir)
            sk = np.mean([l["skill_B3"] for l in linhas
                          if l["modelo"] == melhor and l["skill_B3"] == l["skill_B3"]])
            rob_rows.append({"fonte_chuva": fonte, "horizonte": h, "modelo": melhor,
                             "skill_B3_medio": round(float(sk), 4)})
    rob = pd.DataFrame(rob_rows)
    piv = rob.pivot_table(index="horizonte", columns="fonte_chuva", values="skill_B3_medio")
    piv["dif_power_vs_era5"] = (piv.get("power") - piv.get("era5")).round(4)
    piv["dif_sem1999_vs_power"] = (piv.get("power_sem1999") - piv.get("power")).round(4)
    piv.to_csv(res / "robustez_power.csv", encoding="utf-8")

    # figuras
    logging.info("Figuras...")
    fig_skill_condicional(cond, horizontes, melhor, res / "skill_condicional.png")
    fig_backtest(oof_por_h[90], base, melhor, res / "backtest_2013_16_h90.png")

    # ---- resumo impresso ----
    print("\n" + "=" * 76)
    print("RESUMO Exp 3 — RESIDUAL sobre B3 (validação; teste 2023+ INTOCADO)")
    print("=" * 76)
    print(f"\nMelhor modelo residual: {melhor}")
    print("\n[Residual vs B3] skill_B3 médio (>0 = bate o B3):")
    print(resumo.pivot_table(index="modelo", columns="horizonte", values="skill_B3").to_string())
    print("\n[Skill condicional] skill_B3 (IC95%) por subconjunto × horizonte:")
    for sub in ["geral", "seca", "pre_episodio"]:
        s = cond[(cond.subconjunto == sub) & (cond.modelo == melhor)]
        print(f"  {sub} ({melhor}): " + " | ".join(
            f"h{int(r.horizonte)}={r.skill_B3} [{r.ic95_lo},{r.ic95_hi}] n={int(r.n)}"
            for _, r in s.iterrows()))
    print("\n[Robustez POWER] skill_B3 médio por fonte de chuva:")
    print(piv.round(4).to_string())
    print("\n[Antecedência] (ver antecedencia_exp3.csv) — amostra limiar 40, h=90:")
    amostra = ant[(ant.limiar == 40) & (ant.horizonte == 90)]
    print(amostra[["episodio_inicio", "modelo", "threshold_vol", "primeiro_alerta",
                   "antecedencia_dias", "falso_alarme_episodios_ano"]].to_string(index=False))
    print(f"\nCSVs e figuras em: {res}")
    print("=" * 76)
    logging.info("OK.")


if __name__ == "__main__":
    main()
