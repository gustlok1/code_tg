#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp4_outros_reservatorios.py — Exp 4 (H4): o MESMO pipeline em outros reservatorios.

Demonstra que o arcabouco (B3 + XGBRes residual, folds com janela crescente + purga,
SPI/SPEI no treino) roda em reservatorios distintos, nao so no agregado do Cantareira.

Fonte: os 4 reservatorios INDIVIDUAIS do Sistema Cantareira (Jaguari-Jacarei 808 hm3,
Cachoeira 70 hm3, Atibainha 96 hm3, Paiva Castro 8 hm3), com dinamicas bem diferentes.
Cada um tem serie propria de volume e vazoes (data/raw_v2/sar_cantareira_diario.csv) e
clima proprio no seu ponto (data/raw_v2/openmeteo_pontos_diario.csv). Nenhum download novo.

LIMITACAO (time-box, CLAUDE.md item 5): reservatorios FORA do Cantareira (Nordeste e SIN)
nao entraram porque os endpoints /sar0/Medicao e /sar0/MedicaoSin servem a serie via AJAX
com estado de sessao (o HTML estatico traz so o cabecalho), ao contrario do
/sar0/MedicaoCantareira que e renderizado no servidor. Caminhos para depois: endpoint AJAX
interno, Claude-in-Chrome dirigindo a pagina, ou dados abertos da ONS. Ver DECISOES.md.

Saidas: reports/v2/exp4/ (skill por reservatorio + figura).
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import modelagem_v2 as mv
import transformadores as tr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "v2" / "exp4"
SAR = ROOT / "data" / "raw_v2" / "sar_cantareira_diario.csv"
ERA5 = ROOT / "data" / "raw_v2" / "openmeteo_pontos_diario.csv"
HORIZONTES = [30, 60, 90]
W_ACC = (7, 30, 90, 180, 365)
# codigo SAR -> nome do ponto no arquivo ERA5 (mesma grafia)
RESERVATORIOS = {29001: "JAGUARI-JACAREI", 29002: "CACHOEIRA",
                 29003: "ATIBAINHA", 29004: "PAIVA CASTRO"}
XGB = dict(max_depth=3, n_estimators=300, learning_rate=0.03, reg_lambda=15.0,
           subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
           random_state=42, n_jobs=-1, objective="reg:squarederror")

FEAT_CLIMA = ([f"precip_acc{w}" for w in W_ACC] + [f"et0_acc{w}" for w in W_ACC]
              + [f"bal_acc{w}" for w in W_ACC] + ["tmean_ma7", "tmean_ma30", "doy_sin", "doy_cos"])
FEAT_ESTADO = ["vol_pct", "vol_var7", "vol_var30", "aflu_ma7", "aflu_ma30", "deflu_ma30"]


def _utf8():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass


def montar_base(vol, clima):
    ini = max(vol["data"].min(), clima["data"].min())
    cal = pd.DataFrame({"data": pd.date_range(ini, max(vol["data"].max(), clima["data"].max()), freq="D")})
    b = (cal.merge(vol, on="data", how="left").merge(clima, on="data", how="left")
            .sort_values("data").reset_index(drop=True))
    for w in W_ACC:
        b[f"precip_acc{w}"] = b["precip_mm"].rolling(w, min_periods=w).sum()
        b[f"et0_acc{w}"] = b["et0_mm"].rolling(w, min_periods=w).sum()
        b[f"bal_acc{w}"] = b[f"precip_acc{w}"] - b[f"et0_acc{w}"]
    b["tmean_ma7"] = b["tmean_c"].rolling(7, min_periods=7).mean()
    b["tmean_ma30"] = b["tmean_c"].rolling(30, min_periods=30).mean()
    doy = b["data"].dt.dayofyear; nd = b["data"].dt.is_leap_year.map({True: 366, False: 365})
    b["doy_sin"] = np.sin(2 * np.pi * doy / nd); b["doy_cos"] = np.cos(2 * np.pi * doy / nd)
    b["vol_var7"] = b["vol_pct"] - b["vol_pct"].shift(7)
    b["vol_var30"] = b["vol_pct"] - b["vol_pct"].shift(30)
    b["aflu_ma7"] = b["afluencia"].rolling(7, min_periods=7).mean()
    b["aflu_ma30"] = b["afluencia"].rolling(30, min_periods=30).mean()
    b["deflu_ma30"] = b["defluencia"].rolling(30, min_periods=30).mean()
    for h in HORIZONTES:
        b[f"vol_pct_t{h}"] = b["vol_pct"].shift(-h)
        b[f"vol_var_t{h}"] = b[f"vol_pct_t{h}"] - b["vol_pct"]
    return b


def folds_proporcionais(datas, n=4, frac_treino_min=0.4):
    d0, dN = datas.min(), datas.max()
    bordas = pd.date_range(d0 + (dN - d0) * frac_treino_min, dN, periods=n + 1)
    return [{"nome": f"f{i+1}", "ini": bordas[i].strftime("%Y-%m-%d"),
             "fim": bordas[i + 1].strftime("%Y-%m-%d")} for i in range(n)]


def validar(base, folds):
    res = {}
    for h in HORIZONTES:
        sk_b3, sk_b1 = [], []
        for fo in folds:
            trm, vam = mv.mascaras_fold(base["data"], fo["ini"], fo["fim"], h)
            spi = tr.aplicar_indicadores(base, pd.Series(trm, index=base.index), col_data="data",
                                         col_precip="precip_mm", col_et0="et0_mm", escalas=(3, 6, 12))
            feats = FEAT_CLIMA + list(spi.columns) + FEAT_ESTADO
            X = base[FEAT_CLIMA + FEAT_ESTADO].join(spi)
            alvo = f"vol_pct_t{h}"
            ok = X[feats].notna().all(axis=1).values & base[alvo].notna().values & base["vol_pct"].notna().values
            tr_ok, va_ok = trm & ok, vam & ok
            if tr_ok.sum() < 200 or va_ok.sum() < 30:
                continue
            delta = mv.ajustar_B3(base[tr_ok], h)
            b3_va = mv.prever_B3(base[va_ok], h, delta)
            r_tr = base.loc[tr_ok, alvo].values - mv.prever_B3(base[tr_ok], h, delta)
            m = XGBRegressor(**XGB); m.fit(X[tr_ok], r_tr)
            pred = b3_va + m.predict(X[va_ok])
            real = base.loc[va_ok, alvo].values; b1 = base.loc[va_ok, "vol_pct"].values
            mae = mean_absolute_error(real, pred)
            mae_b3 = mean_absolute_error(real, b3_va); mae_b1 = mean_absolute_error(real, b1)
            sk_b3.append(1 - mae / mae_b3 if mae_b3 > 0 else np.nan)
            sk_b1.append(1 - mae / mae_b1 if mae_b1 > 0 else np.nan)
        res[h] = {"skill_vs_B3": round(float(np.nanmean(sk_b3)), 4) if sk_b3 else np.nan,
                  "skill_vs_B1": round(float(np.nanmean(sk_b1)), 4) if sk_b1 else np.nan,
                  "n_folds": len(sk_b3)}
    return res


def main():
    _utf8()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S")
    OUT.mkdir(parents=True, exist_ok=True)
    sar = pd.read_csv(SAR, parse_dates=["data"])
    era = pd.read_csv(ERA5, parse_dates=["data"])
    linhas = []
    for cod, nome in RESERVATORIOS.items():
        logging.info(f"=== {nome} (cod {cod}) ===")
        vol = (sar[sar["codigo"] == cod][["data", "vol_util_pct", "afluencia_m3s", "defluencia_m3s"]]
               .rename(columns={"vol_util_pct": "vol_pct", "afluencia_m3s": "afluencia",
                                "defluencia_m3s": "defluencia"}))
        clima = (era[era["ponto"] == nome][["data", "precip_mm", "et0_mm", "tmean_c"]])
        if len(vol) < 500 or len(clima) < 500:
            logging.warning(f"  poucos dados ({len(vol)} vol, {len(clima)} clima), pulando")
            continue
        base = montar_base(vol, clima)
        folds = folds_proporcionais(base.dropna(subset=["vol_pct"])["data"])
        res = validar(base, folds)
        vmin = float(base["vol_pct"].min())
        cob = f"{base['data'].min().date()}..{base['data'].max().date()}"
        logging.info(f"  cobertura {cob} | vol_min {vmin:.1f}% | {res}")
        for h in HORIZONTES:
            linhas.append({"reservatorio": nome, "codigo": cod, "horizonte": h,
                           "cobertura": cob, "vol_min_pct": round(vmin, 1), **res[h]})

    tab = pd.DataFrame(linhas)
    tab.to_csv(OUT / "exp4_skill.csv", index=False, encoding="utf-8")

    fig, ax = plt.subplots(figsize=(9, 5))
    for nome in tab.reservatorio.unique():
        d = tab[tab.reservatorio == nome].sort_values("horizonte")
        ax.plot(d.horizonte, d.skill_vs_B3, marker="o", label=nome)
    ax.axhline(0, color="#d62728", ls="--", lw=1, label="B3")
    ax.set_xlabel("Horizonte (dias)"); ax.set_ylabel("skill vs B3 (validacao)")
    ax.set_xticks(HORIZONTES)
    ax.set_title("Exp 4: o mesmo pipeline nos 4 reservatorios do Cantareira (skill vs B3)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "exp4_skill.png", dpi=140); plt.close(fig)

    print("\n" + "=" * 78)
    print("EXP 4 — o mesmo pipeline em reservatorios distintos (validacao)")
    print("=" * 78)
    print(tab.to_string(index=False))
    print(f"\nSaidas: {OUT}")
    print("=" * 78)


if __name__ == "__main__":
    main()
