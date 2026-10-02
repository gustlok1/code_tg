#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_experimento_sintetico.py — Exp 1 (SINTÉTICO). Mostra, em ambiente controlado:
(a) que o pipeline recupera um sinal conhecido; (b) que o ML só supera o B3 quando a
chuva futura tem componente previsível a partir do passado (eixo φ); (c) como o
desempenho varia com o nº de crises N. Calibra φ/σ com o real para marcar onde o
Cantareira está. NÃO abre o teste real; nenhuma mistura com dados reais.

Tudo marcado como sintético: saídas em data/sintetico/ e reports_v2/sintetico/, com
marca d'água "DADOS SINTÉTICOS" em toda figura.

Uso: python pipeline_v2/04_experimento_sintetico.py
"""
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analise_exp3 as ax
import dataset_v2 as dv
import sintetico_v2 as sv

ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path(__file__).resolve().parent / "config.yaml"
REPORTS = ROOT / "reports_v2" / "sintetico"
PHIS = [0.0, 0.5, 0.8, 0.95]
NS = [3, 10, 30]
SEEDS = [0, 1, 2]


def _utf8():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass


def _marca(fig):
    fig.text(0.5, 0.5, "DADOS SINTÉTICOS", fontsize=46, color="gray", alpha=0.16,
             ha="center", va="center", rotation=28, zorder=100)


def _prauc_entrada(oof, limiar=40):
    y = (oof["real"].values < limiar).astype(int)
    ent = oof["vol_t"].values > limiar
    ye = y[ent]
    if ent.sum() == 0 or ye.sum() == 0 or ye.sum() == len(ye):
        return np.nan, (ye.mean() if len(ye) else np.nan)
    return float(average_precision_score(ye, -oof["XGBRes"].values[ent])), float(ye.mean())


# ---------------------------------------------------------------- figuras
def fig_heatmap(grid, phi_real, out):
    piv = (grid[grid.horizonte == 90].groupby(["phi", "N"])["skill_B3"].mean()
           .unstack("N").reindex(index=PHIS, columns=NS))
    fig, ax_ = plt.subplots(figsize=(7, 5))
    im = ax_.imshow(piv.values, aspect="auto", cmap="RdBu_r", vmin=-0.4, vmax=0.4, origin="lower")
    ax_.set_xticks(range(len(NS))); ax_.set_xticklabels(NS)
    ax_.set_yticks(range(len(PHIS))); ax_.set_yticklabels(PHIS)
    ax_.set_xlabel("N (nº de crises)"); ax_.set_ylabel("φ (persistência da chuva)")
    for i in range(len(PHIS)):
        for j in range(len(NS)):
            ax_.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
    # marca o φ real (interpola a posição no eixo categórico)
    pos = np.interp(phi_real, PHIS, range(len(PHIS)))
    ax_.axhline(pos, color="black", lw=2, ls="--")
    ax_.text(len(NS) - 0.4, pos + 0.08, f"φ real ≈ {phi_real:.2f}", color="black", fontsize=9, ha="right")
    fig.colorbar(im, label="skill vs B3 (h=90)")
    ax_.set_title("Skill vs B3 (h=90) — φ × N")
    _marca(fig); fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


def fig_skill_por_phi(grid, horizontes, out, N_fixo=10):
    g = grid[grid.N == N_fixo]
    fig, ax_ = plt.subplots(figsize=(8, 5))
    cores = {30: "#1f77b4", 60: "#ff7f0e", 90: "#2ca02c"}
    for h in horizontes:
        sub = g[g.horizonte == h].groupby("phi")["skill_B3"]
        med = sub.mean().reindex(PHIS); lo = sub.min().reindex(PHIS); hi = sub.max().reindex(PHIS)
        ax_.plot(PHIS, med.values, marker="o", color=cores[h], label=f"h={h}")
        ax_.fill_between(PHIS, lo.values, hi.values, color=cores[h], alpha=0.15)
    ax_.axhline(0, color="#d62728", ls="--", lw=1, label="B3")
    ax_.set_xlabel("φ (persistência da chuva)"); ax_.set_ylabel("skill vs B3")
    ax_.set_title(f"Skill vs B3 por φ (N={N_fixo}; faixa = seeds)"); ax_.legend(); ax_.grid(alpha=.3)
    _marca(fig); fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


def fig_prauc_por_N(grid, out):
    g = grid[grid.horizonte == 90]
    fig, ax_ = plt.subplots(figsize=(8, 5))
    for phi in PHIS:
        sub = g[g.phi == phi].groupby("N")["pr_auc_entrada"].mean().reindex(NS)
        ax_.plot(NS, sub.values, marker="o", label=f"φ={phi}")
    prev = g["prevalencia"].mean()
    ax_.axhline(prev, color="#7f7f7f", ls=":", label=f"prevalência média ≈ {prev:.3f}")
    ax_.set_xlabel("N (nº de crises)"); ax_.set_ylabel("PR-AUC de entrada (limiar 40, h=90)")
    ax_.set_title("PR-AUC de entrada por N (uma linha por φ)"); ax_.legend(); ax_.grid(alpha=.3)
    _marca(fig); fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


def fig_exemplo(base, comp, out):
    fig, axs = plt.subplots(3, 1, figsize=(13, 7), sharex=True)
    axs[0].plot(base["data"], base["era5_precip_mm"], color="#1f77b4", lw=0.5)
    axs[0].set_ylabel("Chuva (mm/d)")
    axs[1].plot(base["data"], base["afluencia_m3s_sistema"], color="#2ca02c", lw=0.7)
    axs[1].set_ylabel("Afluência (m³/s)")
    faixas = [(60, 100, "#2ca02c"), (40, 60, "#bcbd22"), (30, 40, "#ff7f0e"),
              (20, 30, "#d62728"), (-30, 20, "#7f0000")]
    for lo, hi, c in faixas:
        axs[2].axhspan(lo, hi, color=c, alpha=0.10)
    axs[2].plot(base["data"], base["vol_util_pct_sistema"], color="#1f3b73", lw=0.8)
    axs[2].set_ylabel("Volume útil (%)"); axs[2].set_xlabel("Data")
    axs[0].set_title("Série sintética de exemplo (chuva, afluência, volume com faixas)")
    _marca(fig); fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


def fig_barras_sanidade(san, out):
    fig, ax_ = plt.subplots(figsize=(7, 5))
    nomes = ["real (modelo)", "embaralhado", "oráculo\n(diagnóstico)"]
    vals = [san["normal"]["skill"], san["shuffle"]["skill"], san["oraculo"]["skill"]]
    cores = ["#2ca02c", "#7f7f7f", "#9467bd"]
    ax_.bar(nomes, vals, color=cores)
    ax_.axhline(0, color="#d62728", ls="--", lw=1)
    for i, v in enumerate(vals):
        ax_.text(i, v + (0.01 if v >= 0 else -0.03), f"{v:.3f}", ha="center")
    ax_.set_ylabel("skill vs B3 (h=90)")
    ax_.set_title("Sanidade — real vs embaralhado vs oráculo\n(oráculo usa chuva futura: diagnóstico, não é previsão)")
    _marca(fig); fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    _utf8()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S")
    t0 = time.time()
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    REPORTS.mkdir(parents=True, exist_ok=True)
    real = ROOT / cfg["caminhos"]["processed_v2_dataset"]

    logging.info("Calibrando com o real (φ, σ, climatologia)...")
    phi_real, sigma_real = sv.estimar_ar1_real(real)
    clim = sv.calibrar_clima_real(real)
    pd.DataFrame([{"phi_real": phi_real, "sigma_real": sigma_real}]).to_csv(
        REPORTS / "calibracao_real.csv", index=False, encoding="utf-8")
    logging.info(f"φ real = {phi_real} | σ real = {sigma_real}")

    logging.info("Rodando a grade (φ × N × seeds = 36 cenários)...")
    grid_rows, ant_rows = [], []
    exemplo = None
    for phi in PHIS:
        for N in NS:
            for seed in SEEDS:
                nome = f"phi{phi}_N{N}_seed{seed}"
                base, comp = sv.gerar_cenario(phi, N, seed, clim)
                cdir = sv.caminho_cenario(nome); cdir.mkdir(parents=True, exist_ok=True)
                base.to_parquet(cdir / "base.parquet", index=False)
                ds = sv.construir_dataset(base, cfg)
                epis40 = dv.detectar_episodios(base, "vol_util_pct_sistema", 40,
                                               cfg["episodio_gap_max_dias"],
                                               cfg["episodio_dur_min_dias"])
                oof90 = None
                for h in cfg["horizontes"]:
                    sk, oof = sv.modelar_sintetico(ds, cfg, h, "normal")
                    prauc, prev = _prauc_entrada(oof, 40)
                    grid_rows.append({"phi": phi, "N": N, "seed": seed, "horizonte": h,
                                      "skill_B3": round(sk, 4), "pr_auc_entrada": round(prauc, 4) if prauc == prauc else np.nan,
                                      "prevalencia": round(prev, 4) if prev == prev else np.nan,
                                      "n_episodios40": len(epis40)})
                    if h == 90:
                        oof90 = oof
                # antecedência pareada (h=90, limiar 40)
                if oof90 is not None and len(epis40):
                    _, rp = ax.antecedencia_pareada(
                        oof90, base[["data", "vol_util_pct_sistema"]], epis40, 90, 40,
                        modelo_a="XGBRes", modelo_b="B3", n_boot=400, seed=seed)
                    ant_rows.append({"phi": phi, "N": N, "seed": seed,
                                     "mediana_dif_antecedencia": rp["mediana_dif"],
                                     "wilcoxon_p": rp["wilcoxon_p"]})
                if phi == 0.8 and N == 10 and seed == 0:
                    exemplo = (base, comp)
        logging.info(f"  φ={phi} ok ({time.time()-t0:.0f}s)")

    grid = pd.DataFrame(grid_rows)
    grid.to_csv(REPORTS / "grade_sintetico.csv", index=False, encoding="utf-8")
    pd.DataFrame(ant_rows).to_csv(REPORTS / "antecedencia_sintetico.csv", index=False, encoding="utf-8")

    # sanidade (cenário representativo φ=0.8, N=10, seed 0)
    logging.info("Sanidade: embaralhado e oráculo...")
    base_s, _ = sv.gerar_cenario(0.8, 10, 0, clim)
    ds_s = sv.construir_dataset(base_s, cfg)
    san = {}
    for modo in ("normal", "shuffle", "oraculo"):
        sk, oof = sv.modelar_sintetico(ds_s, cfg, 90, modo)
        prauc, prev = _prauc_entrada(oof, 40)
        san[modo] = {"skill": round(sk, 4), "pr_auc_entrada": round(prauc, 4) if prauc == prauc else np.nan,
                     "prevalencia": round(prev, 4) if prev == prev else np.nan}
    pd.DataFrame(san).T.to_csv(REPORTS / "sanidade_sintetico.csv", encoding="utf-8")

    # figuras
    logging.info("Figuras (com marca d'água)...")
    fig_heatmap(grid, phi_real, REPORTS / "heatmap_skill_phi_N.png")
    fig_skill_por_phi(grid, cfg["horizontes"], REPORTS / "skill_por_phi.png")
    fig_prauc_por_N(grid, REPORTS / "prauc_entrada_por_N.png")
    if exemplo:
        fig_exemplo(exemplo[0], exemplo[1], REPORTS / "serie_exemplo.png")
    fig_barras_sanidade(san, REPORTS / "barras_sanidade.png")

    # ---- resumo ----
    print("\n" + "=" * 78)
    print("RESUMO Exp 1 — SINTÉTICO (DADOS SINTÉTICOS; teste real INTOCADO)")
    print("=" * 78)
    print(f"\nCalibração real: φ = {phi_real}  |  σ = {sigma_real}  "
          f"(→ o Cantareira real está em φ ≈ 0)")
    print("\n[Grade] skill vs B3 (h=90), média das seeds (φ linhas × N colunas):")
    piv = grid[grid.horizonte == 90].groupby(["phi", "N"])["skill_B3"].mean().unstack("N").reindex(PHIS, columns=NS)
    print(piv.round(3).to_string())
    print("\n[Grade] PR-AUC de entrada (limiar 40, h=90), média das seeds:")
    piv2 = grid[grid.horizonte == 90].groupby(["phi", "N"])["pr_auc_entrada"].mean().unstack("N").reindex(PHIS, columns=NS)
    print(piv2.round(3).to_string())
    print("\n[Sanidade] (cenário φ=0.8, N=10, h=90):")
    for modo in ("normal", "shuffle", "oraculo"):
        print(f"  {modo:9s}: skill_B3={san[modo]['skill']:+.3f} | "
              f"PR-AUC entrada={san[modo]['pr_auc_entrada']} | prevalência={san[modo]['prevalencia']}")
    print(f"\nTempo total: {time.time()-t0:.0f}s")
    print(f"Figuras/CSVs em: {REPORTS}  |  cenários em: {sv.DIR_SINTETICO}")
    print("=" * 78)


if __name__ == "__main__":
    main()
