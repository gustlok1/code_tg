#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_congelar_e_testar.py — Congela as escolhas da validação e, SÓ depois, abre o teste.

Rigor (CLAUDE.md seção 7): o teste (2023-01-01 em diante) só é aberto uma vez, depois do
congelamento. Fluxo em duas fases:

  1) python pipeline_v2/03_congelar_e_testar.py
     Grava reports_v2/final/congelamento.json (conjuntos, modelos, hiperparâmetros e
     thresholds escolhidos na VALIDAÇÃO, para B3 e XGBRes, limiares 40 e 30). NÃO abre o teste.
     -> em seguida faça commit do congelamento.json.

  2) python pipeline_v2/03_congelar_e_testar.py --abrir-teste
     Lê o congelamento, abre o teste UMA vez (registra em reports_v2/final/teste_aberto.log),
     treina com o treino purgado (< 2023) e avalia no teste (>= 2023). Relatório com destaque
     para a crise de 2025-26. Não altera nenhuma escolha depois de aberto.
"""
import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, mean_absolute_error

import analise_exp3 as ax
import modelagem_v2 as mv

ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path(__file__).resolve().parent / "config.yaml"
FINAL = ROOT / "reports_v2" / "final"
CONGELAMENTO = FINAL / "congelamento.json"


def _utf8():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass


def _features_nome():
    anom = [f"anom_era5_precip_acc{w}" for w in ax.ANOM_W]
    return ax._cols_clima("era5") + list(ax.ESTADO) + list(mv.SPI_COLS) + anom


# ---------------------------------------------------------------- congelamento
def congelar(df, cfg):
    FINAL.mkdir(parents=True, exist_ok=True)
    modelos_xgb, thresholds = {}, {40: {}, 30: {}}
    for h in cfg["horizontes"]:
        escolhidos = ax.melhor_config_residual(df, cfg, h)
        modelos_xgb[str(h)] = {"config": escolhidos["XGBRes"]["config"],
                               "mae_val": escolhidos["XGBRes"]["mae"]}
        _, oof = ax.validar_residual(df, cfg, h, src="era5")
        for limiar in (40, 30):
            crise = pd.Series(oof["real"].values < limiar,
                              index=pd.to_datetime(oof["data"]).values)
            thresholds[limiar][str(h)] = {
                "B3": round(float(ax._threshold_vol(oof, "B3", crise)), 2),
                "XGBRes": round(float(ax._threshold_vol(oof, "XGBRes", crise)), 2)}
        logging.info(f"  congelado h={h}: XGBRes={modelos_xgb[str(h)]['config']}")

    cong = {
        "congelado_em": datetime.now().isoformat(timespec="seconds"),
        "conjunto": "CLIMA_ESTADO + SPI/SPEI(3,6,12) + anomalias sazonais (90,180,365)",
        "features": _features_nome(),
        "alvo": "volume util do sistema (%) em t+h; residual sobre o B3",
        "modelos": {
            "B3": "persistencia + variacao sazonal por dia do ano (sem hiperparametro)",
            "XGBRes": modelos_xgb},
        "thresholds_volume_alerta": thresholds,
        "teste_inicio": cfg["split"]["teste_inicio"],
        "folds_validacao": cfg["split"]["folds"],
        "observacao": "alerta = volume PREVISTO abaixo do threshold; thresholds escolhidos por "
                      "F1 na validacao. Nada muda depois de aberto o teste.",
    }
    CONGELAMENTO.write_text(json.dumps(cong, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + "=" * 76)
    print("CONGELAMENTO gravado (teste ainda INTOCADO)")
    print("=" * 76)
    print(f"Arquivo: {CONGELAMENTO}")
    print(f"Conjunto: {cong['conjunto']}")
    for h in cfg["horizontes"]:
        print(f"  h={h}: XGBRes {cong['modelos']['XGBRes'][str(h)]['config']} | "
              f"thr40(B3/XGB)={thresholds[40][str(h)]['B3']}/{thresholds[40][str(h)]['XGBRes']} | "
              f"thr30(B3/XGB)={thresholds[30][str(h)]['B3']}/{thresholds[30][str(h)]['XGBRes']}")
    print("\nPROXIMO PASSO: faca commit do congelamento.json e depois rode com --abrir-teste.")
    print("=" * 76)


# ---------------------------------------------------------------- abrir teste
def _runs(bool_series):
    return ax._runs(bool_series)


def _antecedencia(alerta, vol_diaria, episodios, h, limiar, cob_ini, cob_fim):
    """Antecedência por episódio e falso-alarme por episódio no período, com threshold FIXO."""
    crise_real = vol_diaria < limiar
    eps = episodios.copy(); eps["ini"] = pd.to_datetime(eps["inicio"])
    eps = eps[(eps["ini"] >= cob_ini) & (eps["ini"] <= cob_fim)]
    # falso-alarme por episodio
    falsos = 0
    for a, b in ax._runs(alerta):
        jc = crise_real[(crise_real.index >= a) & (crise_real.index <= b + pd.Timedelta(days=h + 30))]
        if not bool(jc.any()):
            falsos += 1
    linhas = []
    for _, ep in eps.iterrows():
        ini = ep["ini"]
        jan = alerta[(alerta.index >= ini - pd.Timedelta(days=120)) & (alerta.index < ini)]
        disp = jan[jan]
        ant = int((ini - disp.index.min()).days) if len(disp) else np.nan
        prim = disp.index.min().strftime("%Y-%m-%d") if len(disp) else "sem alerta"
        linhas.append({"episodio_inicio": ini.strftime("%Y-%m-%d"), "fim": ep["fim"],
                       "primeiro_alerta": prim, "antecedencia_dias": ant})
    return linhas, falsos


def abrir_teste(df, cfg):
    if not CONGELAMENTO.exists():
        raise SystemExit("congelamento.json nao existe. Rode primeiro sem --abrir-teste e faca commit.")
    cong = json.loads(CONGELAMENTO.read_text(encoding="utf-8"))
    FINAL.mkdir(parents=True, exist_ok=True)
    with (FINAL / "teste_aberto.log").open("a", encoding="utf-8") as fh:
        fh.write(f"TESTE ABERTO em {datetime.now().isoformat(timespec='seconds')} "
                 f"(teste_inicio={cong['teste_inicio']}; congelado_em={cong['congelado_em']})\n")
    logging.warning("Abrindo o TESTE uma unica vez (registrado em teste_aberto.log)...")

    proc = ROOT / "data" / "processed_v2"
    epis = {30: pd.read_csv(proc / "episodios_crise.csv"),
            40: pd.read_csv(proc / "episodios_crise40.csv")}
    vol_diaria = df.set_index("data")["vol_util_pct_sistema"].sort_index()

    met_rows, ant_rows = [], []
    for h in cfg["horizontes"]:
        config = cong["modelos"]["XGBRes"][str(h)]["config"]
        oof = ax.testar_residual(df, cfg, h, config)
        real = oof["real"].values
        for modelo in ("B3", "XGBRes"):
            pred = oof[modelo].values
            mae = mean_absolute_error(real, pred)
            mae_b3 = mean_absolute_error(real, oof["B3"].values)
            mae_b1 = mean_absolute_error(real, oof["vol_t"].values)
            # PR-AUC de entrada (limiar 40)
            y = (real < 40).astype(int); ent = oof["vol_t"].values > 40
            prauc = (average_precision_score(y[ent], -pred[ent])
                     if ent.sum() and 0 < y[ent].sum() < ent.sum() else np.nan)
            met_rows.append({"horizonte": h, "modelo": modelo, "n": len(real),
                             "MAE": round(float(mae), 3),
                             "skill_B3": round(1 - mae / mae_b3, 4) if mae_b3 > 0 else np.nan,
                             "skill_B1": round(1 - mae / mae_b1, 4) if mae_b1 > 0 else np.nan,
                             "PR_AUC_entrada40": round(float(prauc), 4) if prauc == prauc else np.nan})
            # antecedência por episodio (thresholds congelados), limiares 40 e 30
            datas = pd.to_datetime(oof["data"])
            for limiar in (40, 30):
                thr = cong["thresholds_volume_alerta"][str(limiar)][str(h)][modelo]
                alerta = pd.Series(pred < thr, index=datas.values).sort_index()
                linhas, falsos = _antecedencia(alerta, vol_diaria, epis[limiar], h, limiar,
                                               datas.min(), datas.max())
                for L in linhas:
                    ant_rows.append({"horizonte": h, "modelo": modelo, "limiar": limiar,
                                     "threshold": thr, "falso_alarme_abs_teste": falsos, **L})

    met = pd.DataFrame(met_rows); ant = pd.DataFrame(ant_rows)
    met.to_csv(FINAL / "metricas_teste.csv", index=False, encoding="utf-8")
    ant.to_csv(FINAL / "antecedencia_teste.csv", index=False, encoding="utf-8")

    print("\n" + "=" * 80)
    print("RESULTADO DO TESTE (2023+), modelos CONGELADOS — aberto uma unica vez")
    print("=" * 80)
    print("\n[Regressao do volume] MAE e skill vs B3/B1 no teste:")
    print(met.pivot_table(index="modelo", columns="horizonte",
                          values=["MAE", "skill_B3"]).round(3).to_string())
    print("\n[PR-AUC de entrada (limiar 40)] no teste:")
    print(met.pivot_table(index="modelo", columns="horizonte", values="PR_AUC_entrada40").round(3).to_string())
    print("\n[DESTAQUE: crise de 2025-26] antecedência por modelo (limiar 40):")
    foco = ant[(ant.limiar == 40) & (ant.episodio_inicio >= "2025-01-01")]
    if len(foco):
        print(foco[["episodio_inicio", "horizonte", "modelo", "threshold", "primeiro_alerta",
                    "antecedencia_dias", "falso_alarme_abs_teste"]].to_string(index=False))
    else:
        print("  (nenhum episodio de limiar 40 iniciando em 2025+ no teste)")
    print("\n[Falso-alarme absoluto no teste] (episodios de alarme nao seguidos de crise):")
    fa = ant.groupby(["modelo", "horizonte", "limiar"])["falso_alarme_abs_teste"].first().reset_index()
    print(fa.pivot_table(index=["modelo", "horizonte"], columns="limiar",
                         values="falso_alarme_abs_teste").to_string())
    print(f"\nArquivos: {FINAL}")
    print("=" * 80)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--abrir-teste", action="store_true",
                    help="Abre o teste 2023+ UMA vez (requer congelamento.json commitado).")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    _utf8()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    df = pd.read_parquet(ROOT / cfg["caminhos"]["processed_v2_dataset"])
    df["data"] = pd.to_datetime(df["data"])

    if args.abrir_teste:
        abrir_teste(df, cfg)
    else:
        logging.info("Congelando escolhas da validacao (teste intocado)...")
        congelar(df, cfg)


if __name__ == "__main__":
    main()
