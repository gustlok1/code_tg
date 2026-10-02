#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04b_sintetico_ajuste.py — Ajuste do Exp 1 (sintético), ADITIVO (não apaga a grade).

1. Persistência EFETIVA φ_ef (autocorr lag-1 das anomalias mensais, MESMA função do real)
   para cada cenário da grade existente; refaz as figuras com φ_ef no eixo.
2. Cenários SEM secas (N=0) para φ ∈ {0; 0,5; 0,8; 0,95}, 3 seeds.
3. Gêmeo calibrado: τ ajustado à afluência real, ruído multiplicativo = desvio do resíduo,
   φ e σ reais, N=0, 10 seeds. Responde: com as propriedades do Cantareira real, o ML
   empata com o B3?
4. Figura nova: skill vs B3 (h=90) por φ_ef, pontos por N, gêmeo destacado, faixa real.

NÃO abre o teste real; nenhuma mistura com dados reais no dataset sintético.
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

import dataset_v2 as dv
import sintetico_v2 as sv

ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path(__file__).resolve().parent / "config.yaml"
REPORTS = ROOT / "reports" / "v2" / "sintetico"
RESULT = ROOT / "reports" / "v2" / "resultados"
PHIS = [0.0, 0.5, 0.8, 0.95]
SEEDS3 = [0, 1, 2]


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
        return np.nan
    return float(average_precision_score(ye, -oof["XGBRes"].values[ent]))


def _modelar_cenario(base, cfg):
    """Roda normal p/ h=30/60/90 e devolve dict com skill, pr-auc e φ_ef."""
    ds = sv.construir_dataset(base, cfg)
    phi_ef = sv.phi_lag1_mensal(base["data"], base["era5_precip_mm"])
    out = {"phi_ef": phi_ef}
    for h in cfg["horizontes"]:
        sk, oof = sv.modelar_sintetico(ds, cfg, h, "normal")
        out[f"skill_{h}"] = sk
        out[f"prauc_{h}"] = _prauc_entrada(oof, 40)
    return out


def fig_skill_vs_phief(pontos, twin, phi_real, real_band, out):
    fig, ax_ = plt.subplots(figsize=(9, 5.5))
    cmap = {0: "#7f7f7f", 3: "#1f77b4", 10: "#ff7f0e", 30: "#2ca02c"}
    for N, c in cmap.items():
        d = pontos[pontos.N == N]
        if len(d):
            ax_.scatter(d.phi_ef, d.skill_90, c=c, s=45, alpha=0.8, edgecolor="k",
                        linewidth=0.3, label=f"N={N}")
    # faixa do resultado real (Exp 2/3, h=90)
    ax_.axhspan(real_band[0], real_band[1], color="#d62728", alpha=0.12)
    ax_.axhline(real_band[2], color="#d62728", ls="-", lw=1.2, label="skill real (Exp 2/3, h=90)")
    ax_.axvline(phi_real, color="black", ls="--", lw=1.5, label=f"φ real ≈ {phi_real:.2f}")
    # gêmeo calibrado destacado
    ax_.scatter(twin["phi_ef"], twin["skill_90"], marker="*", s=380, c="#9467bd",
                edgecolor="k", linewidth=0.6, zorder=6, label="gêmeo calibrado")
    ax_.axhline(0, color="#555", lw=0.8, ls=":")
    ax_.set_xlabel("φ_ef (persistência efetiva da chuva)")
    ax_.set_ylabel("skill vs B3 (h=90)")
    ax_.set_title("Skill vs B3 (h=90) por persistência EFETIVA φ_ef")
    ax_.legend(fontsize=8, loc="upper left"); ax_.grid(alpha=0.3)
    _marca(fig); fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


def fig_skill_por_phief_linha(pontos, horizontes, out):
    fig, ax_ = plt.subplots(figsize=(9, 5))
    cores = {30: "#1f77b4", 60: "#ff7f0e", 90: "#2ca02c"}
    for h in horizontes:
        d = pontos.sort_values("phi_ef")
        ax_.scatter(d["phi_ef"], d[f"skill_{h}"], c=cores[h], s=22, alpha=0.6, label=f"h={h}")
    ax_.axhline(0, color="#d62728", ls="--", lw=1, label="B3")
    ax_.set_xlabel("φ_ef"); ax_.set_ylabel("skill vs B3")
    ax_.set_title("Skill vs B3 por φ_ef (todos os cenários)"); ax_.legend(); ax_.grid(alpha=.3)
    _marca(fig); fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


def main():
    _utf8()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S")
    t0 = time.time()
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    real = ROOT / cfg["caminhos"]["processed_v2_dataset"]
    phi_real, sigma_real = sv.estimar_ar1_real(real)
    clim = sv.calibrar_clima_real(real)

    logging.info("Calibrando τ do reservatório linear à afluência real...")
    tau_fit = sv.calibrar_tau_real(real)
    logging.info(f"τ={tau_fit['tau']} d | k={tau_fit['k']} | RMSE={tau_fit['rmse']} m³/s "
                 f"| erro_rel={tau_fit['erro_rel']} | σ_mult={tau_fit['sigma_mult']}")
    pd.DataFrame([tau_fit]).to_csv(REPORTS / "calibracao_tau.csv", index=False, encoding="utf-8")

    # ---- 1. φ_ef na grade existente (regenera precip com o gerador padrão) ----
    pontos_rows = []
    grade = pd.read_csv(REPORTS / "grade_sintetico.csv")
    chaves = grade[["phi", "N", "seed"]].drop_duplicates().itertuples(index=False)
    logging.info("φ_ef da grade existente + skills (reusa os skills já calculados)...")
    g90 = grade[grade.horizonte == 90].set_index(["phi", "N", "seed"])
    for phi, N, seed in [(r.phi, r.N, r.seed) for r in chaves]:
        base, _ = sv.gerar_cenario(phi, N, seed, clim)             # gerador PADRÃO
        phi_ef = sv.phi_lag1_mensal(base["data"], base["era5_precip_mm"])
        row = {"origem": "grade", "phi_param": phi, "N": N, "seed": seed, "phi_ef": round(phi_ef, 4)}
        for h in cfg["horizontes"]:
            sub = grade[(grade.phi == phi) & (grade.N == N) & (grade.seed == seed) & (grade.horizonte == h)]
            row[f"skill_{h}"] = float(sub["skill_B3"].iloc[0]) if len(sub) else np.nan
            row[f"prauc_{h}"] = float(sub["pr_auc_entrada"].iloc[0]) if len(sub) else np.nan
        pontos_rows.append(row)

    # ---- 2. cenários N=0 (só variabilidade natural) ----
    logging.info("Cenários N=0 (sem secas injetadas)...")
    for phi in PHIS:
        for seed in SEEDS3:
            base, _ = sv.gerar_cenario(phi, 0, seed, clim)
            r = _modelar_cenario(base, cfg)
            pontos_rows.append({"origem": "N0", "phi_param": phi, "N": 0, "seed": seed,
                                "phi_ef": round(r["phi_ef"], 4),
                                **{f"skill_{h}": r[f"skill_{h}"] for h in cfg["horizontes"]},
                                **{f"prauc_{h}": r[f"prauc_{h}"] for h in cfg["horizontes"]}})
    pontos = pd.DataFrame(pontos_rows)
    pontos.to_csv(REPORTS / "grade_phief.csv", index=False, encoding="utf-8")

    # ---- 3. gêmeo calibrado (φ real; τ ajustado; ruído na afluência; N=0; 10 seeds) ----
    logging.info("Gêmeo calibrado (10 seeds)...")
    real_mensal = (pd.read_parquet(real, columns=["data", "era5_precip_mm"])
                   .assign(data=lambda d: pd.to_datetime(d["data"])))
    real_mensal = real_mensal[real_mensal.data.dt.year <= 2022]
    rm = real_mensal.set_index("data")["era5_precip_mm"].resample("ME").sum()
    real_mean, real_std = float(rm.mean()), float(rm.std())
    # σ da anomalia calibrado ao DESVIO mensal real (o σ_real do AR1 superdispersa)
    sigma_anom_cal = sv.calibrar_sigma_anom(clim, phi_real, tau_fit["tau"], real_std)
    logging.info(f"σ_anom calibrado = {sigma_anom_cal} (σ_real AR1 = {sigma_real}) | "
                 f"ruído afluência σ_mult = {tau_fit['sigma_mult']}")
    twin_rows, twin_means, twin_stds = [], [], []
    for seed in range(10):
        base, _ = sv.gerar_cenario(phi_real, 0, seed, clim, sigma_anom=sigma_anom_cal,
                                   tau=tau_fit["tau"], ruido_afluencia=tau_fit["sigma_mult"],
                                   normalizar_media=True)
        r = _modelar_cenario(base, cfg)
        tm = base.set_index("data")["era5_precip_mm"].resample("ME").sum()
        twin_means.append(tm.mean()); twin_stds.append(tm.std())
        twin_rows.append({"seed": seed, "phi_ef": round(r["phi_ef"], 4),
                          **{f"skill_{h}": r[f"skill_{h}"] for h in cfg["horizontes"]},
                          **{f"prauc_{h}": r[f"prauc_{h}"] for h in cfg["horizontes"]}})
    twin = pd.DataFrame(twin_rows)
    twin.to_csv(REPORTS / "gemeo_calibrado.csv", index=False, encoding="utf-8")
    twin_resumo = {"phi_ef_medio": round(twin["phi_ef"].mean(), 4),
                   "sigma_anom_cal": sigma_anom_cal, "sigma_mult_afluencia": tau_fit["sigma_mult"],
                   "tau_dias": tau_fit["tau"],
                   "chuva_mensal_media_real": round(real_mean, 2),
                   "chuva_mensal_media_gemeo": round(float(np.mean(twin_means)), 2),
                   "chuva_mensal_std_real": round(real_std, 2),
                   "chuva_mensal_std_gemeo": round(float(np.mean(twin_stds)), 2)}
    for h in cfg["horizontes"]:
        twin_resumo[f"skill_{h}_medio"] = round(float(twin[f"skill_{h}"].mean()), 4)
        twin_resumo[f"skill_{h}_min"] = round(float(twin[f"skill_{h}"].min()), 4)
        twin_resumo[f"skill_{h}_max"] = round(float(twin[f"skill_{h}"].max()), 4)
        twin_resumo[f"prauc_{h}_medio"] = round(float(twin[f"prauc_{h}"].mean()), 4)

    # ---- 4. figuras com φ_ef ----
    # faixa real: do Exp 3 (skill_condicional geral XGBRes h=90), se disponível
    real_band = (-0.02, 0.04, 0.016)
    try:
        sc = pd.read_csv(RESULT / "skill_condicional.csv")
        row = sc[(sc.subconjunto == "geral") & (sc.modelo == "XGBRes") & (sc.horizonte == 90)].iloc[0]
        real_band = (float(row.ic95_lo), float(row.ic95_hi), float(row.skill_B3))
    except Exception:  # noqa: BLE001
        pass
    twin_ponto = {"phi_ef": twin["phi_ef"].mean(), "skill_90": twin["skill_90"].mean()}
    fig_skill_vs_phief(pontos, twin_ponto, phi_real, real_band, REPORTS / "skill_vs_phief.png")
    fig_skill_por_phief_linha(pontos, cfg["horizontes"], REPORTS / "skill_por_phief.png")

    # ---- resumo ----
    print("\n" + "=" * 80)
    print("AJUSTE Exp 1 — persistência EFETIVA φ_ef e gêmeo calibrado (DADOS SINTÉTICOS)")
    print("=" * 80)
    print(f"\nφ real = {phi_real} | σ real = {sigma_real}")
    print(f"τ ajustado = {tau_fit['tau']} d | ganho k = {tau_fit['k']} | "
          f"RMSE = {tau_fit['rmse']} m³/s | erro_rel = {tau_fit['erro_rel']} | "
          f"σ_mult(afluência) = {tau_fit['sigma_mult']}")
    print("\n[φ_ef médio por cenário (φ_param × N)] — note N=0 e o φ do gerador vs efetivo:")
    piv = pontos.groupby(["phi_param", "N"])["phi_ef"].mean().unstack("N")
    print(piv.round(3).to_string())
    print("\n[Skill vs B3 (h=90) por φ_param × N] (inclui N=0):")
    piv2 = pontos.groupby(["phi_param", "N"])["skill_90"].mean().unstack("N")
    print(piv2.round(3).to_string())
    print("\n[Gêmeo calibrado] (φ,σ reais; τ ajustado; N=0; 10 seeds):")
    print(f"  φ_ef médio = {twin_resumo['phi_ef_medio']}  (real = {phi_real})")
    print(f"  chuva mensal média: real={twin_resumo['chuva_mensal_media_real']} "
          f"gêmeo={twin_resumo['chuva_mensal_media_gemeo']} mm | "
          f"std: real={twin_resumo['chuva_mensal_std_real']} gêmeo={twin_resumo['chuva_mensal_std_gemeo']}")
    for h in cfg["horizontes"]:
        print(f"  skill_B3 h{h}: {twin_resumo[f'skill_{h}_medio']:+.3f} "
              f"[{twin_resumo[f'skill_{h}_min']:+.3f}, {twin_resumo[f'skill_{h}_max']:+.3f}] "
              f"| PR-AUC entrada = {twin_resumo[f'prauc_{h}_medio']}")
    pd.DataFrame([twin_resumo]).to_csv(REPORTS / "gemeo_resumo.csv", index=False, encoding="utf-8")
    print(f"\nFaixa real (Exp 2/3, h=90) usada na figura: skill={real_band[2]} [{real_band[0]}, {real_band[1]}]")
    print(f"Tempo: {time.time()-t0:.0f}s | figura nova: {REPORTS/'skill_vs_phief.png'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
