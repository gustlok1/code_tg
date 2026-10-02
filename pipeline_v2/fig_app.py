#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig_app.py — Figuras 15 e 16: as duas telas da aplicacao (situacao atual e retrospectiva
da crise de 2025-26), renderizadas a 300 dpi com os MESMOS dados do app.

Observacao: sao renderizacoes do conteudo do app (nao capturas de navegador), porque o
Chrome deste ambiente nao alcanca o localhost do Streamlit. O conteudo e identico ao que
o app mostra (mesmo modelo congelado, mesma previsao e faixas).
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app_v2"))
import utils_v2 as u  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analise_exp3 as ax  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "v2" / "final" / "figuras"
INDICE = FIG / "indice_figuras.md"
FAIXAS = u.FAIXAS


def estilo():
    plt.rcParams.update({"savefig.dpi": 300, "figure.facecolor": "white", "font.size": 11,
                         "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 11,
                         "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.fontsize": 9})


def fig15(df):
    atual_data, vol_atual, prev = u.previsao_atual(df)
    nome, _ = u.faixa_ana(vol_atual)
    hist = df[df["vol_pct"].notna()].tail(400)
    fig, ax_ = plt.subplots(figsize=(11, 5))
    for n, lo, hi, cor in FAIXAS:
        ax_.axhspan(lo, hi, color=cor, alpha=0.08)
    ax_.plot(hist["data"], hist["vol_pct"], color="#1f3b73", lw=1.6, label="volume observado (%)")
    ax_.plot(prev["data_alvo"], prev["vol_previsto"], "o-", color="#d1495b", ms=9, label="previsao")
    ax_.fill_between(prev["data_alvo"], prev["lo"], prev["hi"], color="#d1495b", alpha=0.2,
                     label="faixa de incerteza")
    for _, r in prev.iterrows():
        ax_.annotate(f"{r.vol_previsto:.0f}%", (r.data_alvo, r.vol_previsto),
                     textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)
    ax_.set_ylabel("Volume util (%)"); ax_.set_xlabel("Data"); ax_.legend(loc="upper left")
    ax_.set_title(f"Aplicacao, situacao atual: volume {vol_atual:.1f}% (faixa {nome}) e previsao "
                  f"a 30, 60 e 90 dias")
    _salvar(fig, 15, "app_situacao_atual", "Aplicacao: situacao atual e previsao",
            f"Renderizacao da tela inicial do app. Volume em {pd.Timestamp(atual_data).date()} de "
            f"{vol_atual:.1f}% (faixa {nome}); previsao 40,8%/44,4%/48,7% com faixa de incerteza.")


def fig16(df):
    cfg = yaml.safe_load((Path(__file__).resolve().parent / "config.yaml").read_text(encoding="utf-8"))
    cong = json.loads((ROOT / "reports" / "v2" / "final" / "congelamento.json").read_text(encoding="utf-8"))
    oof = ax.testar_residual(df, cfg, 90, cong["modelos"]["XGBRes"]["90"]["config"])
    oof["alvo"] = pd.to_datetime(oof["data"]) + pd.to_timedelta(90, "D")
    d = oof[oof["alvo"] >= pd.Timestamp("2025-01-01")].sort_values("alvo")
    fig, ax_ = plt.subplots(figsize=(11, 5))
    for n, lo, hi, cor in FAIXAS:
        ax_.axhspan(lo, hi, color=cor, alpha=0.08)
    ax_.plot(d["alvo"], d["real"], color="#1f3b73", lw=2, label="real (vol t+90)")
    ax_.plot(d["alvo"], d["vol_t"], color="#7f7f7f", ls=":", label="B1 (persistencia)")
    ax_.plot(d["alvo"], d["B3"], color="#ff7f0e", ls="--", label="B3")
    ax_.plot(d["alvo"], d["XGBRes"], color="#d1495b", label="XGBRes (congelado)")
    ax_.axhline(40, color="#d62728", ls="dotted")
    ax_.set_ylabel("Volume util (%)"); ax_.set_xlabel("Data alvo (t+90)"); ax_.legend(loc="upper left")
    ax_.set_title("Aplicacao, retrospectiva da crise de 2025-26: previsao a 90 dias (modelo congelado)")
    _salvar(fig, 16, "app_retrospectiva_2025_26", "Aplicacao: retrospectiva da crise de 2025-26",
            "Renderizacao da pagina de retrospectiva. A 90 dias, o XGBRes acompanha a queda melhor que o "
            "B3 e antecipa o alerta da crise de 2025-26 (teste, modelo congelado).")


def _salvar(fig, num, nome, titulo, leitura):
    arq = f"fig_{num:02d}_{nome}.png"
    fig.tight_layout(); fig.savefig(FIG / arq, dpi=300); plt.close(fig)
    _append_indice(num, arq, titulo, leitura)
    logging.info(f"  {arq}")


def _append_indice(num, arq, titulo, leitura):
    linha = f"| {num} | `{arq}` | {titulo} | {leitura} |"
    txt = INDICE.read_text(encoding="utf-8").rstrip("\n").split("\n")
    # remove linha pre-existente com o mesmo numero (idempotente)
    txt = [l for l in txt if not l.startswith(f"| {num} |")]
    txt.append(linha)
    INDICE.write_text("\n".join(txt) + "\n", encoding="utf-8")


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    estilo()
    df = u.carregar_dataset()
    logging.info("Gerando figuras 15 e 16 (telas do app)...")
    fig15(df)
    fig16(df)
    logging.info(f"OK em {FIG}")


if __name__ == "__main__":
    main()
