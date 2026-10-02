#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp0_diagnostico_v1.py — Exp 0: figuras do diagnóstico da v1 (base do RELATORIO_ESTADO_ATUAL.md).

Mostra, com números vindos dos próprios dados da v1 (modo leitura, sem alterar a v1):
  (a) chuva anual agregada vs número de estações empilhadas (a chuva foi SOMADA);
  (b) RAD_SUM zerada em todo o teste (2020-2024), por troca de grafia da coluna;
  (c) folds do TimeSeriesSplit sem positivos (4 de 5 folds com zero positivos);
  (d) a regra de limiar sobre o SPI contra os modelos da v1 (a regra ganha em AUC-ROC).

Saídas: reports/v2/exp0/ (4 PNG + exp0_numeros.csv).
"""
import glob
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "v2" / "exp0"
DAILY = ROOT / "data" / "features" / "inmet_sp_daily.parquet"
FEAT = ROOT / "data" / "features" / "inmet_sp_daily_features.parquet"
LABELS = ROOT / "data" / "features" / "inmet_sp_daily_labels.parquet"
HOURLY = ROOT / "data" / "interim" / "inmet_sp_hourly_clean.parquet"
COMPARATIVO = ROOT / "reports" / "v2_threshold_otimizado"


def _utf8():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass


def fig_chuva_vs_estacoes(out):
    d = pd.read_parquet(DAILY, columns=["timestamp", "PRECIP_DIARIA", "RAD_SUM"])
    d["ano"] = pd.to_datetime(d["timestamp"]).dt.year
    chuva = d.groupby("ano")["PRECIP_DIARIA"].sum()
    h = pd.read_parquet(HOURLY, columns=["timestamp"])
    h["ano"] = pd.to_datetime(h["timestamp"]).dt.year
    linhas = h.groupby("ano").size()
    horas = h["ano"].map(lambda y: 8784 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 8760)
    est = (h.groupby("ano").size() / h.groupby("ano").apply(
        lambda g: 8784 if (g.name % 4 == 0 and (g.name % 100 != 0 or g.name % 400 == 0)) else 8760,
        include_groups=False)).round(1)
    anos = chuva.index
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.bar(anos, chuva.values, color="#1f77b4", alpha=0.7, label="chuva anual agregada (mm)")
    ax1.set_ylabel("Chuva anual agregada (mm) — SOMA entre estacoes", color="#1f77b4")
    ax1.set_xlabel("Ano")
    ax2 = ax1.twinx()
    ax2.plot(anos, est.reindex(anos).values, color="#d62728", marker="o", lw=2,
             label="nº de estacoes (linhas/hora)")
    ax2.set_ylabel("Nº de estacoes empilhadas", color="#d62728")
    ax1.set_title("v1: a 'chuva' cresce com o NUMERO DE ESTACOES, nao com o clima "
                  "(chuva somada sem identificador)")
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    return chuva, est


def fig_rad_zerada(out):
    d = pd.read_parquet(DAILY, columns=["timestamp", "RAD_SUM"])
    d["ano"] = pd.to_datetime(d["timestamp"]).dt.year
    rad = d.groupby("ano")["RAD_SUM"].mean()
    fig, ax = plt.subplots(figsize=(12, 5))
    cores = ["#d62728" if a >= 2020 else "#2ca02c" for a in rad.index]
    ax.bar(rad.index, rad.values, color=cores)
    ax.set_title("v1: RAD_SUM fica ZERADA em todo o teste (2020-2024) — troca de grafia da coluna "
                 "de radiacao")
    ax.set_ylabel("RAD_SUM media diaria (kJ/m2)"); ax.set_xlabel("Ano")
    ax.axvspan(2019.5, 2024.5, color="#d62728", alpha=0.08)
    ax.text(2022, rad.max() * 0.5, "teste: RAD_SUM = 0", color="#d62728", ha="center", fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    return rad


def fig_folds_sem_positivo(out):
    L = pd.read_parquet(LABELS, columns=["timestamp", "y90"]).sort_values("timestamp")
    L = L.dropna(subset=["y90"]).reset_index(drop=True)
    n = len(L); ntr = int(n * 0.8)
    ytr = L["y90"].iloc[:ntr].astype(int).to_numpy()
    tscv = TimeSeriesSplit(n_splits=5)
    pos = [int(ytr[iva].sum()) for _, iva in tscv.split(np.arange(ntr))]
    fig, ax = plt.subplots(figsize=(9, 5))
    cores = ["#d62728" if p == 0 else "#2ca02c" for p in pos]
    ax.bar([f"fold {i+1}" for i in range(5)], pos, color=cores)
    for i, p in enumerate(pos):
        ax.text(i, p + 0.2, str(p), ha="center")
    ax.set_title("v1: 4 de 5 folds do TimeSeriesSplit tem ZERO positivos "
                 "(threshold degenera, F1 do CV = 0)")
    ax.set_ylabel("positivos no fold de validacao (y90)")
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    return pos


def fig_spi_vs_modelos(out):
    fe = pd.read_parquet(FEAT, columns=["timestamp", "SPI30_APRX"])
    L = pd.read_parquet(LABELS, columns=["timestamp", "y30", "y60", "y90"])
    df = fe.merge(L, on="timestamp").sort_values("timestamp").reset_index(drop=True)
    linhas = []
    for h in [30, 60, 90]:
        d = df.dropna(subset=[f"y{h}"]).reset_index(drop=True)
        n = len(d); ntr = int(n * 0.8)
        te = d.iloc[ntr:]
        y = te[f"y{h}"].astype(int).to_numpy()
        score = -te["SPI30_APRX"].fillna(0).to_numpy()
        auc_spi = roc_auc_score(y, score) if y.sum() > 0 else np.nan
        # AUCs dos modelos da v1 (CSV salvo)
        aucs = {"regra -SPI30": auc_spi}
        csvs = glob.glob(str(COMPARATIVO / f"comparativo_modelos_y{h}_*.csv"))
        if csvs:
            tab = pd.read_csv(sorted(csvs)[-1])
            col = [c for c in tab.columns if "AUC" in c][0]
            for _, r in tab.iterrows():
                aucs[r["Modelo"]] = float(r[col])
        linhas.append((h, aucs))
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    for ax, (h, aucs) in zip(axs, linhas):
        nomes = list(aucs.keys()); vals = [aucs[k] for k in nomes]
        cores = ["#9467bd" if k == "regra -SPI30" else "#1f77b4" for k in nomes]
        ax.bar(nomes, vals, color=cores)
        ax.axhline(0.5, color="#999", ls="--", lw=1)
        ax.set_title(f"y{h}"); ax.set_ylim(0, 1); ax.tick_params(axis="x", rotation=30)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    axs[0].set_ylabel("AUC-ROC (teste)")
    fig.suptitle("v1: uma regra de limiar sobre o SPI30 bate os modelos em AUC-ROC")
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    return linhas


def main():
    _utf8()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S")
    OUT.mkdir(parents=True, exist_ok=True)
    logging.info("Exp 0: gerando figuras do diagnostico da v1...")
    chuva, est = fig_chuva_vs_estacoes(OUT / "chuva_vs_estacoes.png")
    rad = fig_rad_zerada(OUT / "rad_sum_zerada_teste.png")
    pos = fig_folds_sem_positivo(OUT / "folds_sem_positivo.png")
    spi = fig_spi_vs_modelos(OUT / "spi_vs_modelos.png")

    numeros = pd.DataFrame({
        "ano": chuva.index, "chuva_anual_mm": chuva.values,
        "estacoes_aprox": est.reindex(chuva.index).values,
        "rad_sum_media": rad.reindex(chuva.index).values})
    numeros.to_csv(OUT / "exp0_numeros.csv", index=False, encoding="utf-8")

    print("\n" + "=" * 74)
    print("EXP 0 — diagnostico da v1 (figuras)")
    print("=" * 74)
    print(f"Chuva 2003 vs 2019: {chuva.iloc[0]:.0f} mm ({est.iloc[0]} est.) -> "
          f"{chuva.loc[2019]:.0f} mm ({est.loc[2019]} est.)")
    print(f"RAD_SUM media 2019 vs 2021: {rad.loc[2019]:.0f} -> {rad.loc[2021]:.0f} (zerada)")
    print(f"Positivos por fold (y90): {pos}  -> {pos.count(0)} de 5 com zero")
    for h, aucs in spi:
        print(f"y{h} AUC: regra -SPI30={aucs.get('regra -SPI30'):.3f} | "
              + " ".join(f"{k}={v:.3f}" for k, v in aucs.items() if k != 'regra -SPI30'))
    print(f"\nFiguras em: {OUT}")
    print("=" * 74)


if __name__ == "__main__":
    main()
