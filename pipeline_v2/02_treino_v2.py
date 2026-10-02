#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_treino_v2.py — Validação temporal (janela crescente + purga) dos baselines e modelos
de previsão do volume do Cantareira, com SPI/SPEI ajustados dentro de cada fold.

O período de TESTE (2023+) permanece INTOCÁVEL por padrão. Só é avaliado com a flag
--abrir-teste, que grava a data/hora da abertura em reports/v2/resultados/teste_aberto.log.

Uso:
  python pipeline_v2/02_treino_v2.py
  python pipeline_v2/02_treino_v2.py --abrir-teste      # (decisão deliberada; registrada)

Lógica reaproveitável e testada em modelagem_v2.py.
"""
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import modelagem_v2 as mv

ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path(__file__).resolve().parent / "config.yaml"
HORIZ_FIG_MODELO_REG = "XGBReg"
HORIZ_FIG_MODELO_CLF = "XGBClf"


def _utf8():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass


def _num(df, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ---------------------------------------------------------------- figuras
def fig_prev_real(oof_reg_store, conjunto, modelo, h, folds, out):
    o = oof_reg_store[(conjunto, h)][modelo]
    s = pd.DataFrame({"data": pd.to_datetime(o["data"]), "real": o["real"],
                      "pred": o["pred"]}).sort_values("data")
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(s["data"], s["real"], color="#1f3b73", lw=1.0, label="real")
    ax.plot(s["data"], s["pred"], color="#d1495b", lw=0.9, alpha=0.8, label=f"previsto ({modelo})")
    for fo in folds:
        ax.axvline(pd.Timestamp(fo["ini"]), color="#999", ls=":", lw=0.7)
    ax.set_title(f"Validação — previsto × real do volume (t+{h}) — {conjunto} / {modelo}")
    ax.set_ylabel("Volume útil (%)"); ax.set_xlabel("Ano"); ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


def fig_skill(resumo_reg, horizontes, out):
    fig, ax = plt.subplots(figsize=(9, 5))
    series = [("CLIMA", HORIZ_FIG_MODELO_REG, "#1f77b4", "-"),
              ("CLIMA_ESTADO", HORIZ_FIG_MODELO_REG, "#2ca02c", "-"),
              ("CLIMA_ESTADO", "RandomForestReg", "#8c564b", "--"),
              ("CLIMA_ESTADO", "B3_persist_sazonal", "#ff7f0e", ":")]
    for conj, mod, cor, ls in series:
        ys = []
        for h in horizontes:
            sub = resumo_reg[(resumo_reg.conjunto == conj) & (resumo_reg.modelo == mod)
                             & (resumo_reg.horizonte == h)]
            ys.append(sub["skill_B1"].mean() if len(sub) else np.nan)
        ax.plot(horizontes, ys, marker="o", color=cor, ls=ls, label=f"{conj}/{mod}")
    ax.axhline(0, color="#d62728", lw=1, ls="--", label="B1 (persistência)")
    ax.set_title("Skill (1 − MAE/MAE_B1) por horizonte — CLIMA vs CLIMA_ESTADO vs baselines")
    ax.set_xlabel("Horizonte (dias)"); ax.set_ylabel("Skill vs B1"); ax.set_xticks(horizontes)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


def fig_backtest_episodios(base, oof_clf_store, conjunto, modelo, h, limiar, out):
    o = oof_clf_store.get((conjunto, h, limiar), {}).get(modelo)
    if o is None:
        return
    prob = pd.Series(o["prob"], index=pd.to_datetime(o["data"])).sort_index()
    prob = prob[~prob.index.duplicated()]
    vol = base.set_index("data")["vol_util_pct_sistema"]
    episodios = [("2003-04", "2003-01-01", "2005-06-30"),
                 ("2013-16", "2013-06-01", "2016-06-30")]
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
    for ax, (nome, ini, fim) in zip(axes, episodios):
        ini, fim = pd.Timestamp(ini), pd.Timestamp(fim)
        v = vol.loc[ini:fim]
        ax.plot(v.index, v.values, color="#1f3b73", lw=1.1, label="volume (%)")
        ax.axhline(limiar, color="#d62728", ls="--", lw=0.8)
        ax2 = ax.twinx()
        p = prob.loc[(prob.index >= ini) & (prob.index <= fim)]
        ax2.plot(p.index, p.values, color="#d1495b", lw=0.9, alpha=0.8,
                 label=f"P(crise t+{h})")
        ax2.set_ylim(0, 1); ax2.set_ylabel("prob. de crise")
        ax.set_title(f"Backtest {nome} — {conjunto}/{modelo} (validação)")
        ax.set_ylabel("Volume útil (%)")
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--abrir-teste", action="store_true",
                    help="AVALIA o teste 2023+ (decisão deliberada; registrada em log).")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    _utf8()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    res = ROOT / cfg["caminhos"]["resultados_v2"]; res.mkdir(parents=True, exist_ok=True)
    proc = ROOT / "data" / "processed_v2"
    df = pd.read_parquet(ROOT / cfg["caminhos"]["processed_v2_dataset"])
    df["data"] = pd.to_datetime(df["data"])
    horizontes = cfg["horizontes"]; folds = cfg["split"]["folds"]

    logging.info("Validação (janela crescente + purga; SPI/SPEI por fold)...")
    linhas, oof_reg_store, oof_clf_store = [], {}, {}
    for h in horizontes:
        logging.info(f"  horizonte t+{h}")
        spi_folds = []
        for fo in folds:
            trm, _ = mv.mascaras_fold(df["data"], fo["ini"], fo["fim"], h)
            spi_folds.append(mv.spi_spei_do_fold(df, trm))
        for conjunto in mv.CONJUNTOS:
            lr, oof_r = mv.validar_regressao(df, cfg, conjunto, h, spi_folds)
            linhas += lr; oof_reg_store[(conjunto, h)] = oof_r
            for limiar in (40, 30):
                lc, oof_c = mv.validar_classificacao(df, cfg, conjunto, h, limiar, spi_folds)
                linhas += lc; oof_clf_store[(conjunto, h, limiar)] = oof_c

    met = pd.DataFrame(linhas)
    met.to_csv(res / "metricas_validacao.csv", index=False, encoding="utf-8")
    logging.info(f"Métricas salvas: {res / 'metricas_validacao.csv'} ({len(met)} linhas)")

    # resumos (média dos folds)
    met = _num(met, ["MAE", "RMSE", "skill_B1", "skill_B3", "PR_AUC", "ROC_AUC",
                     "Brier", "prevalencia", "n"])
    reg = met[met.tarefa == "regressao"]
    resumo_reg = (reg.groupby(["conjunto", "modelo", "horizonte"], as_index=False)
                     [["MAE", "RMSE", "skill_B1", "skill_B3"]].mean().round(3))
    resumo_reg.to_csv(res / "resumo_regressao.csv", index=False, encoding="utf-8")

    clf = met[met.tarefa == "classificacao"]
    resumo_clf = (clf.groupby(["conjunto", "modelo", "horizonte", "limiar", "recorte"],
                              as_index=False)[["PR_AUC", "ROC_AUC", "Brier", "prevalencia"]]
                     .mean().round(4))
    resumo_clf.to_csv(res / "resumo_classificacao.csv", index=False, encoding="utf-8")

    # antecedência por episódio (OOF, CLIMA_ESTADO, XGBClf, limiar 30)
    episodios = pd.read_csv(proc / "episodios_crise.csv")
    ant = []
    for h in horizontes:
        ant += mv.antecedencia_episodios(oof_clf_store[("CLIMA_ESTADO", h, 30)], episodios,
                                         cfg, "CLIMA_ESTADO", HORIZ_FIG_MODELO_CLF, h, 30)
    ant_df = pd.DataFrame(ant)
    ant_df.to_csv(res / "antecedencia_por_episodio.csv", index=False, encoding="utf-8")

    # figuras
    logging.info("Gerando figuras...")
    base = pd.read_parquet(ROOT / cfg["caminhos"]["processed_v2_dataset"],
                           columns=["data", "vol_util_pct_sistema"])
    base["data"] = pd.to_datetime(base["data"])
    for h in (30, 90):
        fig_prev_real(oof_reg_store, "CLIMA_ESTADO", HORIZ_FIG_MODELO_REG, h, folds,
                      res / f"prev_real_h{h}.png")
    fig_skill(resumo_reg, horizontes, res / "skill_por_horizonte.png")
    fig_backtest_episodios(base, oof_clf_store, "CLIMA_ESTADO", HORIZ_FIG_MODELO_CLF,
                           90, 30, res / "backtest_episodios.png")

    # ---- teste (dormente) ----
    if args.abrir_teste:
        log_path = res / "teste_aberto.log"
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"TESTE ABERTO em {datetime.now().isoformat(timespec='seconds')} "
                     f"(teste_inicio={cfg['split']['teste_inicio']})\n")
        logging.warning(f"Abrindo TESTE (registrado em {log_path})...")
        linhas_te = []
        for h in horizontes:
            for conjunto in mv.CONJUNTOS:
                linhas_te += mv.avaliar_teste(df, cfg, conjunto, h, linhas)
        pd.DataFrame(linhas_te).to_csv(res / "metricas_teste.csv", index=False, encoding="utf-8")
        logging.warning(f"Métricas de teste salvas: {res / 'metricas_teste.csv'}")

    # ---- resumo impresso ----
    print("\n" + "=" * 74)
    print("RESUMO DA VALIDAÇÃO v2 (média dos folds) — teste 2023+ INTOCADO")
    print("=" * 74)
    print("\n[REGRESSÃO] skill vs B1 e B3 (por conjunto/modelo/horizonte):")
    piv = resumo_reg.pivot_table(index=["conjunto", "modelo"], columns="horizonte",
                                 values="skill_B1")
    print("Skill vs B1:\n" + piv.round(3).to_string())
    print(f"\nMAE médio (vol %): min={resumo_reg['MAE'].min():.2f}  "
          f"max={resumo_reg['MAE'].max():.2f}")
    print("\n[CLASSIFICAÇÃO, limiar 40, recorte 'entrada'] PR-AUC médio:")
    c40 = resumo_clf[(resumo_clf.limiar == 40) & (resumo_clf.recorte == "entrada")]
    print(c40.pivot_table(index=["conjunto", "modelo"], columns="horizonte",
                          values="PR_AUC").round(3).to_string())
    print("\n[ANTECEDÊNCIA por episódio] (CLIMA_ESTADO/XGBClf, limiar 30):")
    if len(ant_df):
        print(ant_df[["episodio_inicio", "horizonte", "primeiro_alerta",
                      "antecedencia_dias", "alarmes_falsos_por_ano"]].to_string(index=False))
    print(f"\nFiguras e CSVs em: {res}")
    print("=" * 74)
    logging.info("OK.")


if __name__ == "__main__":
    main()
