#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_build_dataset_v2.py — Constrói o dataset diário v2 (features causais + alvos),
detecta episódios de crise automaticamente e gera o dicionário de dados e as figuras
de EDA. Lê parâmetros de pipeline_v2/config.yaml.

A lógica reaproveitável (e testada) está em dataset_v2.py; aqui fica a orquestração,
as saídas e as figuras. SPI/SPEI (que dependem de climatologia) NÃO entram aqui —
vivem em transformadores.py, para serem ajustados dentro de cada fold no treino.

Uso:
  python pipeline_v2/01_build_dataset_v2.py
Saídas:
  data/processed_v2/dataset_diario.parquet
  data/processed_v2/dicionario_de_dados.csv
  data/processed_v2/episodios_crise.csv            (limiar 30)
  data/processed_v2/episodios_crise40.csv          (limiar 40)
  data/processed_v2/buracos_calendario.csv
  data/processed_v2/ausentes_por_ano.csv
  reports/v2/eda/{volume_faixas.png, chuva_anual_fontes.png, ausentes_por_ano.png}
"""
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import matplotlib
matplotlib.use("Agg")  # backend não-interativo (evita erro Tk em headless)
import matplotlib.pyplot as plt

import dataset_v2 as dv

ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path(__file__).resolve().parent / "config.yaml"


def _utf8():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass


def log():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S")


# ----------------------------------------------------------------------------
# dicionário de dados
# ----------------------------------------------------------------------------
def _descrever(col: str) -> dict:
    """Mapeia um nome de coluna para (descrição, unidade, fonte, janela, causal)."""
    jan = {"7": "7 d", "30": "30 d", "90": "90 d", "180": "180 d", "365": "365 d"}
    fonte_clima = {"era5": "ERA5 (Open-Meteo)", "power": "NASA POWER"}

    if col == "data":
        return dict(descricao="Data (calendário diário contínuo)", unidade="data",
                    fonte="—", janela="—", causal="sim")

    # estado bruto do reservatório / sistema (SAR)
    bruto = {
        "vol_util_hm3_sistema": ("Volume útil do sistema", "hm³"),
        "capacidade_hm3_sistema": ("Capacidade útil do sistema", "hm³"),
        "vol_util_pct_sistema": ("Volume útil do sistema", "%"),
        "afluencia_m3s_sistema": ("Afluência total do sistema", "m³/s"),
        "defluencia_m3s_sistema": ("Defluência total do sistema", "m³/s"),
        "n_reservatorios": ("Nº de reservatórios com dado no dia", "contagem"),
    }
    if col in bruto:
        desc, un = bruto[col]
        return dict(descricao=desc, unidade=un, fonte="ANA/SAR", janela="instantâneo",
                    causal="sim")

    # clima instantâneo da bacia (média dos pontos)
    m = re.match(r"(era5|power)_(precip_mm|et0_mm|tmean_c|tmax_c|tmin_c|rad_mj|rh_pct|wind_ms|pressure_kpa)$", col)
    if m:
        src, var = m.groups()
        nomes = {"precip_mm": ("Chuva diária (média da bacia)", "mm"),
                 "et0_mm": ("Evapotranspiração de referência ET0", "mm"),
                 "tmean_c": ("Temperatura média", "°C"),
                 "tmax_c": ("Temperatura máxima", "°C"),
                 "tmin_c": ("Temperatura mínima", "°C"),
                 "rad_mj": ("Radiação de ondas curtas", "MJ/m²"),
                 "rh_pct": ("Umidade relativa", "%"),
                 "wind_ms": ("Velocidade do vento 2 m", "m/s"),
                 "pressure_kpa": ("Pressão à superfície", "kPa")}
        d, u = nomes[var]
        return dict(descricao=f"{d} — média dos pontos", unidade=u,
                    fonte=fonte_clima[src], janela="1 d", causal="sim")

    # acumulações e balanços
    m = re.match(r"et0_acc(\d+)$", col)
    if m:
        return dict(descricao="ET0 acumulada", unidade="mm", fonte="ERA5 (Open-Meteo)",
                    janela=jan[m.group(1)], causal="sim")
    m = re.match(r"(era5|power)_precip_acc(\d+)$", col)
    if m:
        src, w = m.groups()
        return dict(descricao="Chuva acumulada", unidade="mm", fonte=fonte_clima[src],
                    janela=jan[w], causal="sim")
    m = re.match(r"(era5|power)_bal_acc(\d+)$", col)
    if m:
        src, w = m.groups()
        return dict(descricao="Balanço hídrico acumulado (chuva − ET0; ET0 do ERA5)",
                    unidade="mm", fonte=f"{fonte_clima[src]} + ERA5(ET0)",
                    janela=jan[w], causal="sim")
    m = re.match(r"(era5|power)_tmean_ma(\d+)$", col)
    if m:
        src, w = m.groups()
        return dict(descricao="Temperatura média (média móvel)", unidade="°C",
                    fonte=fonte_clima[src], janela=jan[w], causal="sim")

    # estado do reservatório derivado
    reserv = {
        "vol_pct": ("Volume útil do sistema em t", "%", "instantâneo"),
        "vol_var7": ("Variação do volume útil em 7 dias", "p.p.", "7 d"),
        "vol_var30": ("Variação do volume útil em 30 dias", "p.p.", "30 d"),
        "aflu_ma7": ("Afluência média", "m³/s", "7 d"),
        "aflu_ma30": ("Afluência média", "m³/s", "30 d"),
        "aflu_ma90": ("Afluência média", "m³/s", "90 d"),
        "deflu_ma30": ("Defluência média", "m³/s", "30 d"),
    }
    if col in reserv:
        d, u, w = reserv[col]
        return dict(descricao=d, unidade=u, fonte="ANA/SAR", janela=w, causal="sim")

    if col in ("doy_sin", "doy_cos"):
        return dict(descricao=f"Dia do ano ({'seno' if col.endswith('sin') else 'cosseno'})",
                    unidade="—", fonte="calendário", janela="1 d", causal="sim")
    if col == "pos_2017":
        return dict(descricao="Indicador: vigência da Resolução ANA/DAEE 925 (≥ 2017-05-29)",
                    unidade="0/1", fonte="regulatório", janela="—", causal="sim")

    # alvos
    m = re.match(r"vol_pct_t(\d+)$", col)
    if m:
        return dict(descricao=f"ALVO: volume útil do sistema em t+{m.group(1)}", unidade="%",
                    fonte="ANA/SAR", janela=f"t+{m.group(1)}", causal="não (futuro)")
    m = re.match(r"vol_var_t(\d+)$", col)
    if m:
        return dict(descricao=f"ALVO: variação de volume de t a t+{m.group(1)}", unidade="p.p.",
                    fonte="ANA/SAR", janela=f"t+{m.group(1)}", causal="não (futuro)")
    m = re.match(r"crise_(\d+)$", col)
    if m:
        return dict(descricao=f"ALVO: crise (vol<30%) em t+{m.group(1)}", unidade="0/1",
                    fonte="ANA/SAR", janela=f"t+{m.group(1)}", causal="não (futuro)")
    m = re.match(r"crise40_(\d+)$", col)
    if m:
        return dict(descricao=f"ALVO: crise (vol<40%) em t+{m.group(1)}", unidade="0/1",
                    fonte="ANA/SAR", janela=f"t+{m.group(1)}", causal="não (futuro)")

    return dict(descricao="(sem descrição)", unidade="—", fonte="—", janela="—", causal="?")


def montar_dicionario(cols) -> pd.DataFrame:
    linhas = [{"coluna": c, **_descrever(c)} for c in cols]
    return pd.DataFrame(linhas, columns=["coluna", "descricao", "unidade", "fonte",
                                         "janela", "causal"])


# ----------------------------------------------------------------------------
# figuras
# ----------------------------------------------------------------------------
def fig_volume_faixas(base, episodios, faixas, out):
    fig, ax = plt.subplots(figsize=(14, 5))
    for fx in faixas:
        ax.axhspan(fx["min"], fx["max"], color=fx["cor"], alpha=0.12)
        ax.text(base["data"].min(), (fx["min"] + fx["max"]) / 2, f" {fx['nome']}",
                va="center", ha="left", fontsize=8, color=fx["cor"])
    ax.plot(base["data"], base["vol_util_pct_sistema"], color="#1f3b73", lw=0.9)
    ax.axhline(30, color="#d62728", lw=0.8, ls="--")
    for _, ep in episodios.iterrows():
        ax.axvspan(pd.Timestamp(ep["inicio"]), pd.Timestamp(ep["fim"]),
                   color="#d62728", alpha=0.18)
    ax.set_title("Volume útil do Sistema Cantareira (1984–2026) — faixas ANA/DAEE e episódios (<30%)")
    ax.set_ylabel("Volume útil (%)"); ax.set_xlabel("Ano")
    ax.margins(x=0.01)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


def fig_chuva_anual(base, out):
    d = base.copy(); d["ano"] = d["data"].dt.year
    anual = d.groupby("ano").agg(
        ERA5=("era5_precip_mm", "sum"), POWER=("power_precip_mm", "sum"),
        n=("era5_precip_mm", "size"),
        n_e=("era5_precip_mm", "count"), n_p=("power_precip_mm", "count"))
    # só anos com cobertura razoável em ambas as fontes
    anual = anual[(anual["n_e"] >= 0.9 * anual["n"]) & (anual["n_p"] >= 0.9 * anual["n"])]
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(anual)); w = 0.4
    ax.bar(x - w / 2, anual["ERA5"], w, label="ERA5 (Open-Meteo)", color="#1f77b4")
    ax.bar(x + w / 2, anual["POWER"], w, label="NASA POWER", color="#ff7f0e")
    ax.set_xticks(x); ax.set_xticklabels(anual.index, rotation=90, fontsize=7)
    ax.set_title("Chuva anual média da bacia do Cantareira — ERA5 vs NASA POWER")
    ax.set_ylabel("Chuva anual (mm)"); ax.set_xlabel("Ano"); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


def tabela_ausentes_por_ano(base) -> pd.DataFrame:
    d = base.copy(); d["ano"] = d["data"].dt.year
    checar = {"SAR": "vol_util_pct_sistema", "ERA5": "era5_precip_mm", "POWER": "power_precip_mm"}
    out = pd.DataFrame(index=sorted(d["ano"].unique()))
    for nome, col in checar.items():
        if col in d.columns:
            out[nome] = d.groupby("ano")[col].apply(lambda s: round(100 * s.isna().mean(), 1))
    out.index.name = "ano"
    return out


def fig_tabela_ausentes(tab, out):
    fig, ax = plt.subplots(figsize=(6, 0.28 * len(tab) + 1.2))
    ax.axis("off")
    t = ax.table(cellText=tab.values, rowLabels=tab.index, colLabels=tab.columns,
                 cellLoc="center", loc="center")
    t.auto_set_font_size(False); t.set_fontsize(8); t.scale(1, 1.1)
    ax.set_title("Percentual de dados ausentes por ano e fonte (%)", fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


# ----------------------------------------------------------------------------
def main():
    _utf8(); log()
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cam = cfg["caminhos"]
    proc = ROOT / cam["processed_v2"]; eda = ROOT / cam["reports_eda"]
    proc.mkdir(parents=True, exist_ok=True); eda.mkdir(parents=True, exist_ok=True)

    logging.info("Carregando fontes...")
    sistema = dv.carregar_sistema(ROOT / cam["sar_sistema"])
    era5 = dv.media_bacia(ROOT / cam["era5_pontos"], "era5", dv.ERA5_VARS)
    power = dv.media_bacia(ROOT / cam["power_pontos"], "power", dv.POWER_VARS)

    logging.info("Montando calendário contínuo e base...")
    base, buracos = dv.construir_calendario(sistema, era5, power, cfg["ano_inicial"])
    buracos.to_csv(proc / "buracos_calendario.csv", index=False, encoding="utf-8")

    logging.info("Construindo features causais e alvos...")
    feats = dv.construir_features(base)
    alvos = dv.construir_alvos(base, cfg["horizontes"], cfg["limiar_crise"],
                               cfg["limiar_crise_sensibilidade"])

    dataset = base.merge(feats, on="data").merge(alvos, on="data")
    dataset = dataset.sort_values("data").reset_index(drop=True)
    out_parquet = proc / "dataset_diario.parquet"
    dataset.to_parquet(out_parquet, index=False)

    dic = montar_dicionario(dataset.columns)
    dic.to_csv(proc / "dicionario_de_dados.csv", index=False, encoding="utf-8")

    # episódios (limiares 30 e 40)
    epi30 = dv.detectar_episodios(base, "vol_util_pct_sistema", cfg["limiar_crise"],
                                  cfg["episodio_gap_max_dias"], cfg["episodio_dur_min_dias"])
    epi40 = dv.detectar_episodios(base, "vol_util_pct_sistema", cfg["limiar_crise_sensibilidade"],
                                  cfg["episodio_gap_max_dias"], cfg["episodio_dur_min_dias"])
    epi30.to_csv(proc / "episodios_crise.csv", index=False, encoding="utf-8")
    epi40.to_csv(proc / "episodios_crise40.csv", index=False, encoding="utf-8")

    # figuras
    logging.info("Gerando figuras de EDA...")
    fig_volume_faixas(base, epi30, cfg["faixas_ana_daee"], eda / "volume_faixas.png")
    fig_chuva_anual(base, eda / "chuva_anual_fontes.png")
    tab = tabela_ausentes_por_ano(base)
    tab.to_csv(proc / "ausentes_por_ano.csv", encoding="utf-8")
    fig_tabela_ausentes(tab, eda / "ausentes_por_ano.png")

    # --- resumo + mínimo histórico do volume ---
    idx_min = base["vol_util_pct_sistema"].idxmin()
    vmin = base.loc[idx_min, "vol_util_pct_sistema"]
    dmin = base.loc[idx_min, "data"]
    aus_total = 100 * dataset.drop(columns=["data"]).isna().mean().mean()

    print("\n" + "=" * 70)
    print("RESUMO DO DATASET v2 — data/processed_v2/dataset_diario.parquet")
    print("=" * 70)
    print(f"Período       : {dataset['data'].min().date()} a {dataset['data'].max().date()}")
    print(f"Linhas (dias) : {len(dataset):,}")
    print(f"Colunas       : {dataset.shape[1]} "
          f"(features + alvos + brutos; ver dicionario_de_dados.csv)")
    print(f"Ausentes médio: {aus_total:.1f}% das células (fora a coluna data)")
    print(f"Vol. mínimo   : {vmin:.2f}% em {dmin.date()}")
    print(f"Episódios <30%: {len(epi30)}  |  Episódios <40%: {len(epi40)}")
    print(f"Figuras       : {eda}")
    print("=" * 70)
    logging.info("OK.")


if __name__ == "__main__":
    main()
