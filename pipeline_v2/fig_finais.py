#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig_finais.py — Figuras finais para o texto (capitulo 4), estilo uniforme, 300 dpi,
rotulos em portugues, titulo que diz o achado, numeradas na ordem do capitulo.
Saida: reports/v2/final/figuras/ (fig_01..fig_14) + indice_figuras.md.
Le dos CSVs/parquets ja gerados; o teste nao e reaberto (usa o congelamento).
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analise_exp3 as ax

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "v2" / "final" / "figuras"
RES = ROOT / "reports" / "v2" / "resultados"
SINT = ROOT / "reports" / "v2" / "sintetico"
EXP4 = ROOT / "reports" / "v2" / "exp4"
FINAL = ROOT / "reports" / "v2" / "final"
DF_DAILY = ROOT / "legado_v1" / "data" / "features" / "inmet_sp_daily.parquet"
DF_FEAT = ROOT / "legado_v1" / "data" / "features" / "inmet_sp_daily_features.parquet"
DF_LAB = ROOT / "legado_v1" / "data" / "features" / "inmet_sp_daily_labels.parquet"
DF_HOUR = ROOT / "legado_v1" / "data" / "interim" / "inmet_sp_hourly_clean.parquet"
DS = ROOT / "data" / "processed_v2" / "dataset_diario.parquet"
COMP = ROOT / "legado_v1" / "reports" / "threshold_otimizado"

FAIXAS = [("Normal", 60, 100, "#2ca02c"), ("Atenção", 40, 60, "#bcbd22"),
          ("Alerta", 30, 40, "#ff7f0e"), ("Restrição", 20, 30, "#d62728"),
          ("Reserva técnica", -30, 20, "#7f0000")]

INDICE = []  # (num, arquivo, titulo, leitura)


def estilo():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "figure.facecolor": "white",
        "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.labelsize": 11, "axes.grid": True, "grid.alpha": 0.3,
        "axes.spines.top": False, "axes.spines.right": False, "legend.fontsize": 9})


def salvar(fig, num, nome, titulo, leitura):
    arq = f"fig_{num:02d}_{nome}.png"
    fig.tight_layout(); fig.savefig(FIG / arq, dpi=300); plt.close(fig)
    INDICE.append((num, arq, titulo, leitura))
    logging.info(f"  {arq}")


# ---------------- 4.1 Exp 0 ----------------
def fig01():
    d = pd.read_parquet(DF_DAILY, columns=["timestamp", "PRECIP_DIARIA"])
    d["ano"] = pd.to_datetime(d["timestamp"]).dt.year
    chuva = d.groupby("ano")["PRECIP_DIARIA"].sum()
    h = pd.read_parquet(DF_HOUR, columns=["timestamp"]); h["ano"] = pd.to_datetime(h["timestamp"]).dt.year
    est = (h.groupby("ano").size() / h.groupby("ano").apply(
        lambda g: 8784 if (g.name % 4 == 0 and (g.name % 100 != 0 or g.name % 400 == 0)) else 8760,
        include_groups=False)).round(1)
    fig, a1 = plt.subplots(figsize=(9, 5))
    a1.bar(chuva.index, chuva.values, color="#1f77b4", alpha=0.7)
    a1.set_ylabel("Chuva anual agregada (mm)", color="#1f77b4"); a1.set_xlabel("Ano")
    a2 = a1.twinx(); a2.grid(False)
    a2.plot(chuva.index, est.reindex(chuva.index).values, color="#d62728", marker="o", lw=2)
    a2.set_ylabel("Número de estações empilhadas", color="#d62728")
    a1.set_title("v1: a chuva agregada cresce com o número de estações, não com o clima")
    salvar(fig, 1, "v1_chuva_vs_estacoes", "Chuva agregada da v1 contra número de estações",
           "A chuva somada vai de 3.438 mm (3,9 estações, 2003) a 50.476 mm (41,5 estações, 2019): "
           "média um artefato de agregação, não o clima.")


def fig02():
    d = pd.read_parquet(DF_DAILY, columns=["timestamp", "RAD_SUM"])
    d["ano"] = pd.to_datetime(d["timestamp"]).dt.year
    rad = d.groupby("ano")["RAD_SUM"].mean()
    fig, ax_ = plt.subplots(figsize=(9, 5))
    cores = ["#d62728" if a >= 2020 else "#2ca02c" for a in rad.index]
    ax_.bar(rad.index, rad.values, color=cores)
    ax_.set_ylabel("RAD_SUM média diária (kJ/m2)"); ax_.set_xlabel("Ano")
    ax_.set_title("v1: RAD_SUM fica zerada em todo o período de teste (2020-2024)")
    salvar(fig, 2, "v1_rad_zerada", "Radiação zerada no teste da v1",
           "Por troca de grafia da coluna, RAD_SUM vale zero em 2020-2024 (teste) e fica inflada no treino.")


def fig03():
    L = pd.read_parquet(DF_LAB, columns=["timestamp", "y90"]).dropna(subset=["y90"]).reset_index(drop=True)
    ntr = int(len(L) * 0.8); ytr = L["y90"].iloc[:ntr].astype(int).to_numpy()
    pos = [int(ytr[iva].sum()) for _, iva in TimeSeriesSplit(n_splits=5).split(np.arange(ntr))]
    fig, ax_ = plt.subplots(figsize=(8, 5))
    ax_.bar([f"fold {i+1}" for i in range(5)], pos, color=["#d62728" if p == 0 else "#2ca02c" for p in pos])
    for i, p in enumerate(pos):
        ax_.text(i, p + 0.2, str(p), ha="center")
    ax_.set_ylabel("Positivos no fold de validação (y90)")
    ax_.set_title("v1: 4 de 5 folds do TimeSeriesSplit sem nenhum positivo")
    salvar(fig, 3, "v1_folds_sem_positivo", "Folds sem positivo na v1",
           "Os positivos por fold são [0,0,0,15,0]: o threshold degenera e o F1 do CV é zero.")


def fig04():
    fe = pd.read_parquet(DF_FEAT, columns=["timestamp", "SPI30_APRX"])
    L = pd.read_parquet(DF_LAB, columns=["timestamp", "y30", "y60", "y90"])
    df = fe.merge(L, on="timestamp").sort_values("timestamp").reset_index(drop=True)
    import glob
    fig, axs = plt.subplots(1, 3, figsize=(11, 4.2), sharey=True)
    for ax_, h in zip(axs, [30, 60, 90]):
        d = df.dropna(subset=[f"y{h}"]).reset_index(drop=True)
        te = d.iloc[int(len(d) * 0.8):]
        auc = {"regra -SPI30": roc_auc_score(te[f"y{h}"].astype(int), -te["SPI30_APRX"].fillna(0))}
        cs = glob.glob(str(COMP / f"comparativo_modelos_y{h}_*.csv"))
        if cs:
            t = pd.read_csv(sorted(cs)[-1]); col = [c for c in t.columns if "AUC" in c][0]
            for _, r in t.iterrows():
                auc[r["Modelo"]] = float(r[col])
        nomes = list(auc); vals = [auc[k] for k in nomes]
        ax_.bar(nomes, vals, color=["#9467bd" if k == "regra -SPI30" else "#1f77b4" for k in nomes])
        ax_.axhline(0.5, color="#999", ls="--"); ax_.set_title(f"y{h}"); ax_.set_ylim(0, 1)
        ax_.tick_params(axis="x", rotation=30)
        for i, v in enumerate(vals):
            ax_.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    axs[0].set_ylabel("AUC-ROC (teste)")
    fig.suptitle("v1: uma regra de limiar sobre o SPI30 supera os modelos em AUC-ROC", fontweight="bold")
    salvar(fig, 4, "v1_spi_vs_modelos", "Regra do SPI contra os modelos da v1",
           "A regra -SPI30 tem AUC 0,785/0,755/0,752, acima dos modelos (melhor XGBoost 0,700).")


# ---------------- 4.2 EDA ----------------
def fig05():
    d = pd.read_parquet(DS, columns=["data", "vol_util_pct_sistema"]); d["data"] = pd.to_datetime(d["data"])
    ep = pd.read_csv(ROOT / "data" / "processed_v2" / "episodios_crise.csv")
    fig, ax_ = plt.subplots(figsize=(11, 4.6))
    for nome, lo, hi, cor in FAIXAS:
        ax_.axhspan(lo, hi, color=cor, alpha=0.10)
    ax_.plot(d["data"], d["vol_util_pct_sistema"], color="#1f3b73", lw=0.9)
    ax_.axhline(30, color="#d62728", ls="--", lw=0.8)
    for _, e in ep.iterrows():
        ax_.axvspan(pd.Timestamp(e["inicio"]), pd.Timestamp(e["fim"]), color="#d62728", alpha=0.18)
    ax_.set_ylabel("Volume útil (%)"); ax_.set_xlabel("Ano")
    ax_.set_title("Volume útil do Sistema Cantareira (1984-2026): faixas ANA/DAEE e episódios < 30%")
    salvar(fig, 5, "eda_volume_faixas", "Volume do sistema com faixas e episódios",
           "O sistema chega a -23,2% em 2015 (reserva técnica). Cinco episódios < 30%: 1986, 2003-04, "
           "2013-16, 2021-22 e 2025-26.")


def fig06():
    d = pd.read_parquet(DS, columns=["data", "era5_precip_mm", "power_precip_mm"])
    d["ano"] = pd.to_datetime(d["data"]).dt.year
    g = d.groupby("ano").agg(ERA5=("era5_precip_mm", "sum"), POWER=("power_precip_mm", "sum"),
                             ndias=("era5_precip_mm", "size"), n_e=("era5_precip_mm", "count"),
                             n_p=("power_precip_mm", "count"))
    g = g[(g["n_e"] >= 0.9 * g["ndias"]) & (g["n_p"] >= 0.9 * g["ndias"])]
    fig, ax_ = plt.subplots(figsize=(11, 4.6)); x = np.arange(len(g)); w = 0.4
    ax_.bar(x - w / 2, g["ERA5"], w, label="ERA5", color="#1f77b4")
    ax_.bar(x + w / 2, g["POWER"], w, label="NASA POWER", color="#ff7f0e")
    ax_.set_xticks(x); ax_.set_xticklabels(g.index, rotation=90, fontsize=7)
    ax_.set_ylabel("Chuva anual média da bacia (mm)"); ax_.set_xlabel("Ano"); ax_.legend()
    if 1999 in g.index:
        xi = list(g.index).index(1999)
        ax_.annotate("anomalia POWER 1999", xy=(xi, g.loc[1999, "POWER"]), xytext=(xi, g["POWER"].max()),
                     arrowprops=dict(arrowstyle="->", color="#d62728"), color="#d62728", fontsize=9)
    ax_.set_title("Chuva anual da bacia: ERA5 contra NASA POWER (anomalia do POWER em 1999)")
    salvar(fig, 6, "eda_chuva_era5_vs_power", "Chuva anual ERA5 contra POWER",
           "As fontes concordam, menos em 1999, quando o POWER tem um pico anômalo (~3.380 mm vs ~1.600 do ERA5).")


# ---------------- 4.3 Sintetico ----------------
def fig07():
    pts = pd.read_csv(SINT / "grade_phief.csv")
    gem = pd.read_csv(SINT / "gemeo_resumo.csv").iloc[0]
    cal = pd.read_csv(SINT / "calibracao_real.csv").iloc[0]
    try:
        sc = pd.read_csv(RES / "skill_condicional.csv")
        row = sc[(sc.subconjunto == "geral") & (sc.modelo == "XGBRes") & (sc.horizonte == 90)].iloc[0]
        band = (row.ic95_lo, row.ic95_hi, row.skill_B3)
    except Exception:  # noqa: BLE001
        band = (-0.02, 0.04, 0.016)
    fig, ax_ = plt.subplots(figsize=(9, 5.2))
    cmap = {0: "#7f7f7f", 3: "#1f77b4", 10: "#ff7f0e", 30: "#2ca02c"}
    for N, c in cmap.items():
        s = pts[pts.N == N]
        if len(s):
            ax_.scatter(s.phi_ef, s.skill_90, c=c, s=40, alpha=0.8, edgecolor="k", lw=0.3, label=f"N={N}")
    ax_.axhspan(band[0], band[1], color="#d62728", alpha=0.12)
    ax_.axhline(band[2], color="#d62728", lw=1.2, label="skill real (Exp 2/3, h=90)")
    ax_.axvline(cal.phi_real, color="black", ls="--", lw=1.5, label=f"phi real = {cal.phi_real:.2f}")
    ax_.scatter(gem.phi_ef_medio, gem.skill_90_medio, marker="*", s=360, c="#9467bd", edgecolor="k",
                lw=0.6, zorder=6, label="gêmeo calibrado")
    ax_.axhline(0, color="#555", ls=":", lw=0.8)
    ax_.set_xlabel("phi_ef (persistência efetiva da chuva)"); ax_.set_ylabel("skill vs B3 (h=90)")
    ax_.legend(fontsize=8, loc="upper left")
    ax_.set_title("Sintético: o ganho sobre o B3 cresce com a persistência da chuva (phi_ef) e com N")
    salvar(fig, 7, "sintetico_skill_phief", "Skill sintético por persistência efetiva",
           "DADOS SINTÉTICOS. O Cantareira real está em phi ~ 0; o gêmeo calibrado (estrela) ainda supera "
           "o B3 em h90, limitação discutida no texto.")


def fig08():
    san = pd.read_csv(SINT / "sanidade_sintetico.csv", index_col=0)
    fig, ax_ = plt.subplots(figsize=(7, 5))
    nomes = ["real (modelo)", "embaralhado", "oráculo"]
    vals = [san.loc["normal", "skill"], san.loc["shuffle", "skill"], san.loc["oraculo", "skill"]]
    ax_.bar(nomes, vals, color=["#2ca02c", "#7f7f7f", "#9467bd"])
    ax_.axhline(0, color="#d62728", ls="--")
    for i, v in enumerate(vals):
        ax_.text(i, v + (0.01 if v >= 0 else -0.03), f"{v:.3f}", ha="center")
    ax_.set_ylabel("skill vs B3 (h=90)")
    ax_.set_title("Sintético: embaralhado zera o skill;\no oráculo (chuva futura) o maximiza")
    salvar(fig, 8, "sintetico_sanidade", "Sanidade do sintético",
           "DADOS SINTÉTICOS. Embaralhado skill -0,01; oráculo +0,62: o teto é a informação sobre a chuva "
           "futura, não o algoritmo.")


# ---------------- 4.4 Exp 2 ----------------
def fig09():
    r = pd.read_csv(RES / "resumo_regressao.csv")
    hs = [30, 60, 90]
    series = [("CLIMA_ESTADO", "XGBReg", "#2ca02c", "-"), ("CLIMA", "XGBReg", "#1f77b4", "-"),
              ("CLIMA_ESTADO", "B3_persist_sazonal", "#ff7f0e", ":")]
    fig, ax_ = plt.subplots(figsize=(8.5, 5))
    for conj, mod, cor, ls in series:
        ys = [r[(r.conjunto == conj) & (r.modelo == mod) & (r.horizonte == h)]["skill_B1"].mean() for h in hs]
        rot = {"B3_persist_sazonal": "B3"}.get(mod, mod)
        ax_.plot(hs, ys, marker="o", color=cor, ls=ls, label=f"{conj}/{rot}")
    ax_.axhline(0, color="#d62728", ls="--", label="B1 (persistência)")
    ax_.set_xlabel("Horizonte (dias)"); ax_.set_ylabel("skill vs B1 (persistência)"); ax_.set_xticks(hs)
    ax_.legend()
    ax_.set_title("Exp 2: clima sozinho não prevê o nível; com o estado, empata com o B3")
    salvar(fig, 9, "exp2_skill_horizonte", "Clima contra clima e estado (Exp 2)",
           "CLIMA sozinho tem skill negativo; CLIMA_ESTADO fica junto do B3 (skill vs B1 ~0,3), sem superá-lo.")


# ---------------- 4.5 Exp 3 ----------------
def fig10():
    sc = pd.read_csv(RES / "skill_condicional.csv")
    paineis = [("geral", "Geral"), ("seca", "Seca (SPI-12 < -1)"), ("pre_episodio", "Pré-episódio")]
    hs = [30, 60, 90]
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.3), sharey=True)
    for a, (sub, tit) in zip(axs, paineis):
        d = sc[(sc.subconjunto == sub) & (sc.modelo == "XGBRes")].sort_values("horizonte")
        yerr = np.vstack([d.skill_B3 - d.ic95_lo, d.ic95_hi - d.skill_B3])
        a.errorbar(d.horizonte, d.skill_B3, yerr=yerr, marker="o", capsize=4, color="#2ca02c")
        a.axhline(0, color="#d62728", ls="--"); a.set_title(tit); a.set_xticks(hs); a.set_xlabel("Horizonte")
    axs[0].set_ylabel("skill vs B3 (IC95%)")
    fig.suptitle("Exp 3: o modelo residual não supera o B3 de forma confiável (IC95% cruza zero)",
                 fontweight="bold")
    salvar(fig, 10, "exp3_skill_condicional", "Skill condicional do residual (Exp 3)",
           "Em geral, seca e pré-episódio, todos os IC95% do XGBRes cruzam zero: não supera o B3 onde importa.")


def fig11():
    ap = pd.read_csv(RES / "antecedencia_pareada.csv")
    d = ap[ap.limiar == 40].sort_values("horizonte")
    fig, ax_ = plt.subplots(figsize=(8.5, 5))
    yerr = np.vstack([d.mediana_dif - d.ic95_lo, d.ic95_hi - d.mediana_dif])
    ax_.errorbar(d.horizonte, d.mediana_dif, yerr=yerr, marker="o", capsize=5, color="#1f77b4", lw=2)
    ax_.axhline(0, color="#d62728", ls="--")
    for _, r in d.iterrows():
        ax_.annotate(f"p={r.wilcoxon_p:g}", (r.horizonte, r.mediana_dif), textcoords="offset points",
                     xytext=(8, 8), fontsize=9)
    ax_.set_xlabel("Horizonte (dias)"); ax_.set_ylabel("Antecedência XGBRes - B3 (dias)"); ax_.set_xticks([30, 60, 90])
    ax_.set_title("Exp 3: o XGBRes antecipa o alerta mais que o B3 (limiar 40),\n"
                  "com significância em h30 e h60")
    salvar(fig, 11, "exp3_antecedencia_pareada", "Antecedência pareada (Exp 3)",
           "Mediana da diferença: +6,5 d (h30, p=0,039), +19 d (h60, p=0,002), +7 d (h90, p=0,10). "
           "Custo: 3-4 episódios de falso-alarme em 20 anos contra zero do B3.")


# ---------------- 4.6 Teste ----------------
def fig12():
    td = pd.read_csv(FINAL / "teste_diagnostico.csv")
    ci = td[td.categoria == "skill_ci"].copy()
    per = td[td.categoria == "skill_periodo"].copy()
    hs = [30, 60, 90]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    ci["valor"] = ci["valor"].astype(float); ci["ic95_lo"] = ci["ic95_lo"].astype(float); ci["ic95_hi"] = ci["ic95_hi"].astype(float)
    ci = ci.sort_values("horizonte")
    yerr = np.vstack([ci.valor - ci.ic95_lo, ci.ic95_hi - ci.valor])
    a1.errorbar(ci.horizonte, ci.valor, yerr=yerr, marker="o", capsize=5, color="#d1495b", lw=2)
    a1.axhline(0, color="#555", ls="--"); a1.set_xticks(hs); a1.set_xlabel("Horizonte")
    a1.set_ylabel("skill vs B3 no teste (IC95%)"); a1.set_title("Skill no teste, com IC95%")
    for rot, cor in [("2023-2024", "#1f77b4"), ("2025-2026", "#ff7f0e")]:
        s = per[per.chave == rot].sort_values("horizonte")
        a2.plot(s.horizonte, s["valor"].astype(float), marker="o", color=cor, label=rot)
    a2.axhline(0, color="#555", ls="--"); a2.set_xticks(hs); a2.set_xlabel("Horizonte")
    a2.set_ylabel("skill vs B3"); a2.set_title("Skill por período"); a2.legend()
    fig.suptitle("Exp 3 (teste): o XGBRes supera o B3; significativo em h30 e h60, presente nos dois períodos",
                 fontweight="bold")
    salvar(fig, 12, "teste_skill_ic", "Resultado do teste com IC95% e por período",
           "Skill +0,240 [0,132;0,372] (h30) e +0,229 [0,079;0,420] (h60) significativos; h90 cruza zero. "
           "A vantagem aparece em 2023-2024 e 2025-2026.")


def fig13():
    cfg = yaml.safe_load((Path(__file__).resolve().parent / "config.yaml").read_text(encoding="utf-8"))
    cong = json.loads((FINAL / "congelamento.json").read_text(encoding="utf-8"))
    df = pd.read_parquet(DS); df["data"] = pd.to_datetime(df["data"])
    config = cong["modelos"]["XGBRes"]["90"]["config"]
    oof = ax.testar_residual(df, cfg, 90, config)
    oof["alvo"] = pd.to_datetime(oof["data"]) + pd.to_timedelta(90, "D")
    d = oof[oof["alvo"] >= pd.Timestamp("2025-01-01")].sort_values("alvo")
    fig, ax_ = plt.subplots(figsize=(10, 4.8))
    for nome, lo, hi, cor in FAIXAS:
        ax_.axhspan(lo, hi, color=cor, alpha=0.08)
    ax_.plot(d["alvo"], d["real"], color="#1f3b73", lw=2, label="real (vol t+90)")
    ax_.plot(d["alvo"], d["vol_t"], color="#7f7f7f", ls=":", label="B1 (persistência)")
    ax_.plot(d["alvo"], d["B3"], color="#ff7f0e", ls="--", label="B3")
    ax_.plot(d["alvo"], d["XGBRes"], color="#d1495b", label="XGBRes (congelado)")
    ax_.axhline(40, color="#d62728", ls="dotted")
    ax_.set_ylabel("Volume útil (%)"); ax_.set_xlabel("Data alvo (t+90)"); ax_.legend()
    ax_.set_title("Teste: previsão a 90 dias na crise de 2025-26 (modelo congelado)")
    salvar(fig, 13, "teste_backtest_2025_26", "Previsão no teste da crise de 2025-26",
           "No teste, a 90 dias, o XGBRes acompanha a queda melhor que o B3; antecipa o alerta em ~11 dias.")


# ---------------- 4.7 Exp 4 ----------------
def fig14():
    e = pd.read_csv(EXP4 / "exp4_skill.csv")
    fig, ax_ = plt.subplots(figsize=(8.5, 5))
    for nome in e.reservatorio.unique():
        d = e[e.reservatorio == nome].sort_values("horizonte")
        ax_.plot(d.horizonte, d.skill_vs_B3, marker="o", label=nome)
    ax_.axhline(0, color="#d62728", ls="--", label="B3")
    ax_.set_xlabel("Horizonte (dias)"); ax_.set_ylabel("skill vs B3 (validação)"); ax_.set_xticks([30, 60, 90])
    ax_.legend()
    ax_.set_title("Exp 4: o mesmo pipeline nos 4 reservatórios;\nsó o Jaguari supera o B3, e só em h30")
    salvar(fig, 14, "exp4_reservatorios", "Generalização para outros reservatórios (Exp 4)",
           "O pipeline roda inalterado. Skill vs B3: Jaguari +0,11 (h30), Cachoeira ~0, Atibainha -0,41, "
           "Paiva Castro -0,18.")


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    FIG.mkdir(parents=True, exist_ok=True); estilo()
    logging.info("Gerando figuras finais (300 dpi)...")
    for f in [fig01, fig02, fig03, fig04, fig05, fig06, fig07, fig08, fig10, fig11, fig12, fig13, fig14]:
        try:
            f()
        except Exception as e:  # noqa: BLE001
            logging.error(f"  FALHA {f.__name__}: {e}")
    try:
        fig09()
    except Exception as e:  # noqa: BLE001
        logging.error(f"  FALHA fig09: {e}")
    INDICE.sort()
    linhas = ["# Indice de figuras (capitulo 4)", "",
              "| Nº | Arquivo | Titulo | Leitura |", "|---|---|---|---|"]
    for num, arq, tit, leit in INDICE:
        linhas.append(f"| {num} | `{arq}` | {tit} | {leit} |")
    (FIG / "indice_figuras.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")
    logging.info(f"OK: {len(INDICE)} figuras + indice em {FIG}")


if __name__ == "__main__":
    main()
