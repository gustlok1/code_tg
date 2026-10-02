#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03b_teste_diagnostico.py — Diagnósticos PÓS-HOC do teste (2023+).

TODOS os números aqui são PÓS-HOC e NÃO ALTERAM o modelo congelado (congelamento.json).
O teste já foi aberto pelo 03; estes diagnósticos apenas caracterizam o resultado.

Produz (reports_v2/final/teste_diagnostico.csv):
  - IC95% do skill vs B3 no teste, por horizonte (bootstrap em blocos de 90 dias);
  - skill vs B3 separado em 2023-2024 e 2025-2026;
  - antecedência e falso-alarme no teste, limiares 40 e 30 (inclui o episódio jun-set/2026);
  - diagnóstico do B3: um B3 ajustado SÓ em 2017-2022 (pós-Resolução 925) e quanto da vantagem
    do XGBRes no teste ele explica (o B3 congelado usa 1984-2022, misturando regras pré e pós-2017).
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error

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


def _b3_2017(df, cfg, h, oof):
    """B3 diagnóstico: variação sazonal ajustada SÓ em 2017-2022 (pós-Resolução 925),
    com purga (t+h < teste_inicio). Previsto nos mesmos t do oof de teste."""
    ti = pd.Timestamp(cfg["split"]["teste_inicio"])
    m = (df["data"].dt.year.between(2017, 2022) & ((df["data"] + pd.to_timedelta(h, "D")) < ti)
         & df[f"vol_var_t{h}"].notna() & df[f"vol_pct_t{h}"].notna())
    delta = mv.ajustar_B3(df[m], h)
    base = pd.DataFrame({"data": pd.to_datetime(oof["data"]), "vol_pct": oof["vol_t"].values})
    return mv.prever_B3(base, h, delta)


def _antecedencia(alerta, vol_diaria, episodios, h, limiar, cob_ini, cob_fim):
    crise_real = vol_diaria < limiar
    eps = episodios.copy(); eps["ini"] = pd.to_datetime(eps["inicio"])
    eps = eps[(eps["ini"] >= cob_ini) & (eps["ini"] <= cob_fim)]
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
        linhas.append((ini.strftime("%Y-%m-%d"), ant))
    return linhas, falsos


def main():
    _utf8()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S")
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cong = json.loads(CONGELAMENTO.read_text(encoding="utf-8"))
    df = pd.read_parquet(ROOT / cfg["caminhos"]["processed_v2_dataset"]); df["data"] = pd.to_datetime(df["data"])
    proc = ROOT / "data" / "processed_v2"
    epis = {30: pd.read_csv(proc / "episodios_crise.csv"),
            40: pd.read_csv(proc / "episodios_crise40.csv")}
    vol_diaria = df.set_index("data")["vol_util_pct_sistema"].sort_index()
    rng = np.random.default_rng(0)
    linhas = []

    def add(cat, h, chave, valor, lo="", hi="", obs="pos-hoc, nao altera o modelo congelado"):
        linhas.append({"categoria": cat, "horizonte": h, "chave": chave, "valor": valor,
                       "ic95_lo": lo, "ic95_hi": hi, "obs": obs})

    for h in cfg["horizontes"]:
        config = cong["modelos"]["XGBRes"][str(h)]["config"]
        oof = ax.testar_residual(df, cfg, h, config)
        real = oof["real"].values
        err_xgb = np.abs(real - oof["XGBRes"].values)
        err_b3 = np.abs(real - oof["B3"].values)
        datas = pd.to_datetime(oof["data"])

        # (1) IC95% do skill vs B3 (bootstrap blocos de 90 dias)
        dfloat = datas.values.astype("datetime64[D]").astype(float)
        base, lo, hi, N = ax._skill_ci(dfloat, err_xgb, err_b3, rng, n_boot=2000, bloco=90)
        add("skill_ci", h, "XGBRes_vs_B3", base, lo, hi, f"n={N}; pos-hoc")

        # (2) skill separado por periodo
        for rotulo, anos in [("2023-2024", {2023, 2024}), ("2025-2026", {2025, 2026})]:
            sel = datas.dt.year.isin(anos).values
            if sel.sum() > 5 and err_b3[sel].mean() > 0:
                sk = 1 - err_xgb[sel].mean() / err_b3[sel].mean()
                add("skill_periodo", h, rotulo, round(float(sk), 4), obs=f"n={int(sel.sum())}; pos-hoc")

        # (3) antecedencia e falso-alarme (limiares 40 e 30)
        for limiar in (40, 30):
            for modelo in ("B3", "XGBRes"):
                thr = cong["thresholds_volume_alerta"][str(limiar)][str(h)][modelo]
                alerta = pd.Series(oof[modelo].values < thr, index=datas.values).sort_index()
                ant, falsos = _antecedencia(alerta, vol_diaria, epis[limiar], h, limiar,
                                            datas.min(), datas.max())
                add("falso_alarme", h, f"{modelo}_limiar{limiar}", falsos, obs="episodios; pos-hoc")
                for ep_ini, a in ant:
                    add("antecedencia", h, f"{modelo}_limiar{limiar}_{ep_ini}",
                        a if a == a else "sem alerta", obs="dias; pos-hoc")

        # (4) diagnostico do B3 (ajustado so em 2017-2022)
        b3_2017 = _b3_2017(df, cfg, h, oof)
        mae_b3f = mean_absolute_error(real, oof["B3"].values)
        mae_b317 = mean_absolute_error(real, b3_2017)
        mae_xgb = mean_absolute_error(real, oof["XGBRes"].values)
        vant = mae_b3f - mae_xgb                      # vantagem do XGBRes (pp de MAE) sobre B3 congelado
        expl = mae_b3f - mae_b317                     # quanto o B3-2017 ja recupera
        frac = (expl / vant) if vant > 0 else np.nan
        add("b3_diagnostico", h, "MAE_B3_congelado_1984_2022", round(float(mae_b3f), 3), obs="pos-hoc")
        add("b3_diagnostico", h, "MAE_B3_2017_2022", round(float(mae_b317), 3), obs="pos-hoc")
        add("b3_diagnostico", h, "MAE_XGBRes", round(float(mae_xgb), 3), obs="pos-hoc")
        add("b3_diagnostico", h, "vantagem_XGBRes_sobre_B3congelado_pp", round(float(vant), 3), obs="pos-hoc")
        add("b3_diagnostico", h, "explicado_por_B3_2017_pp", round(float(expl), 3), obs="pos-hoc")
        add("b3_diagnostico", h, "fracao_da_vantagem_explicada_por_B3_2017",
            round(float(frac), 3) if frac == frac else "", obs="pos-hoc")

    diag = pd.DataFrame(linhas)
    diag.to_csv(FINAL / "teste_diagnostico.csv", index=False, encoding="utf-8")

    print("\n" + "=" * 80)
    print("DIAGNOSTICO POS-HOC DO TESTE (nao altera o modelo congelado)")
    print("=" * 80)
    print("\n[Skill vs B3 no teste, IC95% (bootstrap blocos 90 d)]:")
    for _, r in diag[diag.categoria == "skill_ci"].iterrows():
        print(f"  h{r.horizonte}: {r.valor} [{r.ic95_lo}, {r.ic95_hi}]")
    print("\n[Skill vs B3 por periodo]:")
    sp = diag[diag.categoria == "skill_periodo"]
    for h in cfg["horizontes"]:
        s = sp[sp.horizonte == h]
        print(f"  h{h}: " + " | ".join(f"{r.chave}={r.valor}" for _, r in s.iterrows()))
    print("\n[Diagnostico do B3 (ajustado so 2017-2022) por horizonte]:")
    bd = diag[diag.categoria == "b3_diagnostico"]
    for h in cfg["horizontes"]:
        s = bd[bd.horizonte == h].set_index("chave")["valor"]
        print(f"  h{h}: MAE B3cong={s.get('MAE_B3_congelado_1984_2022')} "
              f"B3_2017={s.get('MAE_B3_2017_2022')} XGB={s.get('MAE_XGBRes')} | "
              f"vantagem_XGB={s.get('vantagem_XGBRes_sobre_B3congelado_pp')} pp, "
              f"explicada por B3_2017={s.get('fracao_da_vantagem_explicada_por_B3_2017')}")
    print(f"\nArquivo: {FINAL/'teste_diagnostico.csv'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
