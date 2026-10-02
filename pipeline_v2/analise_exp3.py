# -*- coding: utf-8 -*-
"""
analise_exp3.py — Núcleo (importável) da análise do Exp 3 (modelo RESIDUAL sobre o B3).

Adiciona-se ao que já existe (não altera modelagem_v2.py nem transformadores.py):
  - modelo residual: alvo r_h = vol(t+h) − B3(t); previsão final = B3(t) + r_h previsto;
  - features CLIMA_ESTADO + SPI/SPEI (3/6/12) + anomalias sazonais de chuva acumulada
    (90/180/365), tudo ajustado SÓ no treino de cada fold;
  - avaliação condicional (seca por SPI-12<−1; pré-episódio) com IC 95% por bootstrap
    em blocos de 90 dias;
  - antecedência por episódio com B3 ao lado, e falso-alarme medido em EPISÓDIOS/ano;
  - robustez trocando a chuva ERA5 pela POWER (com e sem 1999).

Reaproveita modelagem_v2 (folds, purga, B3) e transformadores (SPI/SPEI, anomalias).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

import modelagem_v2 as mv
import transformadores as tr

W_ACC = (7, 30, 90, 180, 365)
ANOM_W = (90, 180, 365)
ESTADO = ["vol_pct", "vol_var7", "vol_var30", "aflu_ma7", "aflu_ma30", "aflu_ma90",
          "deflu_ma30", "pos_2017"]
SEED = 42

GRADE_RES = {
    "RidgeRes": [{"alpha": 10.0}, {"alpha": 100.0}],
    "XGBRes": [{"max_depth": 2, "n_estimators": 400, "learning_rate": 0.03,
                "reg_lambda": 10.0, "subsample": 0.8, "colsample_bytree": 0.8,
                "min_child_weight": 5},
               {"max_depth": 3, "n_estimators": 400, "learning_rate": 0.03,
                "reg_lambda": 20.0, "subsample": 0.8, "colsample_bytree": 0.8,
                "min_child_weight": 5}],
}


def _cols_clima(src):
    return ([f"{src}_precip_acc{w}" for w in W_ACC]
            + [f"et0_acc{w}" for w in W_ACC]
            + [f"{src}_bal_acc{w}" for w in W_ACC]
            + ["era5_tmean_ma7", "era5_tmean_ma30", "doy_sin", "doy_cos"])


def _criar_res(nome, params):
    if nome == "RidgeRes":
        return Pipeline([("sc", StandardScaler()), ("m", Ridge(**params))])
    return XGBRegressor(random_state=SEED, n_jobs=-1, objective="reg:squarederror", **params)


# ============================================================ preparo do fold
def prep_fold(df, fold, h, src="era5", excluir_anos=()):
    trm, vam = mv.mascaras_fold(df["data"], fold["ini"], fold["fim"], h)
    if excluir_anos:
        trm = trm & (~df["data"].dt.year.isin(list(excluir_anos)).values)

    spi = tr.aplicar_indicadores(df, pd.Series(trm, index=df.index), col_data="data",
                                 col_precip=f"{src}_precip_mm", col_et0="era5_et0_mm",
                                 escalas=(3, 6, 12))
    anom = tr.anomalias_sazonais(df, trm, [f"{src}_precip_acc{w}" for w in ANOM_W])
    clima = _cols_clima(src)
    feat_cols = clima + ESTADO + list(spi.columns) + list(anom.columns)
    X = df[clima + ESTADO].join(spi).join(anom)

    ok = X[feat_cols].notna().all(axis=1).values
    alvo = f"vol_pct_t{h}"
    tr_ok = trm & ok & df[alvo].notna().values
    va_ok = vam & ok & df[alvo].notna().values

    delta = mv.ajustar_B3(df[tr_ok], h)
    b3_tr = mv.prever_B3(df[tr_ok], h, delta)
    b3_va = mv.prever_B3(df[va_ok], h, delta)
    return dict(
        Xtr=X.loc[tr_ok, feat_cols], Xva=X.loc[va_ok, feat_cols],
        r_tr=df.loc[tr_ok, alvo].values - b3_tr,
        real_tr=df.loc[tr_ok, alvo].values, b3_tr=b3_tr,       # p/ teste de reconstrução
        real_va=df.loc[va_ok, alvo].values, b3_va=b3_va,
        b1_va=df.loc[va_ok, "vol_pct"].values, vol_t_va=df.loc[va_ok, "vol_pct"].values,
        spi12_va=spi.loc[va_ok, "spi_12"].values,
        datas_va=df.loc[va_ok, "data"].values, fold=fold["nome"])


# ============================================================ validação residual
def validar_residual(df, cfg, h, src="era5", excluir_anos=()):
    """Valida B3, RidgeRes e XGBRes (residual sobre B3). Retorna (linhas, oof) onde
    oof é um DataFrame de validação (2003-2022) com previsões de volume por modelo."""
    folds = mv.folds_config(cfg)
    preps = [prep_fold(df, fo, h, src, excluir_anos) for fo in folds]

    def _mae(real, pred):
        return mean_absolute_error(real, pred)

    linhas = []
    # B3 baseline
    for p in preps:
        mae = _mae(p["real_va"], p["b3_va"]); mae_b1 = _mae(p["real_va"], p["b1_va"])
        linhas.append(_linha(p["fold"], "B3", h, src, mae,
                             1 - mae / mae_b1 if mae_b1 > 0 else np.nan, 0.0))

    oof = {"data": [], "real": [], "vol_t": [], "spi12": [], "b3": []}
    preds_modelo = {}
    for nome, grade in GRADE_RES.items():
        melhor = {"mae": np.inf, "config": None, "preds": None}
        for params in grade:
            preds, maes = [], []
            for p in preps:
                mod = _criar_res(nome, params)
                mod.fit(p["Xtr"], p["r_tr"])
                pred_vol = p["b3_va"] + mod.predict(p["Xva"])     # residual + B3
                preds.append(pred_vol); maes.append(_mae(p["real_va"], pred_vol))
            mm = float(np.mean(maes))
            if mm < melhor["mae"]:
                melhor = {"mae": mm, "config": params, "preds": preds}
        preds_modelo[nome] = melhor
        for p, pred in zip(preps, melhor["preds"]):
            mae = _mae(p["real_va"], pred)
            mae_b1 = _mae(p["real_va"], p["b1_va"]); mae_b3 = _mae(p["real_va"], p["b3_va"])
            linhas.append(_linha(p["fold"], nome, h, src, mae,
                                 1 - mae / mae_b1 if mae_b1 > 0 else np.nan,
                                 1 - mae / mae_b3 if mae_b3 > 0 else np.nan,
                                 config=str(melhor["config"])))

    # OOF (concatena os folds)
    for i, p in enumerate(preps):
        oof["data"].append(pd.to_datetime(p["datas_va"]))
        oof["real"].append(p["real_va"]); oof["vol_t"].append(p["vol_t_va"])
        oof["spi12"].append(p["spi12_va"]); oof["b3"].append(p["b3_va"])
    oof_df = pd.DataFrame({
        "data": np.concatenate([np.asarray(x) for x in oof["data"]]),
        "real": np.concatenate(oof["real"]), "vol_t": np.concatenate(oof["vol_t"]),
        "spi12": np.concatenate(oof["spi12"]), "B3": np.concatenate(oof["b3"]),
    })
    for nome, melhor in preds_modelo.items():
        oof_df[nome] = np.concatenate(melhor["preds"])
    oof_df = oof_df.sort_values("data").reset_index(drop=True)
    return linhas, oof_df


def _linha(fold, modelo, h, src, mae, skill_b1, skill_b3, config="—"):
    return {"fold": fold, "modelo": modelo, "horizonte": h, "chuva": src,
            "MAE": round(float(mae), 3),
            "skill_B1": round(float(skill_b1), 4) if np.isfinite(skill_b1) else np.nan,
            "skill_B3": round(float(skill_b3), 4) if np.isfinite(skill_b3) else np.nan,
            "config": config}


# ============================================================ congelamento/teste residual
def melhor_config_residual(df, cfg, h, src="era5") -> dict:
    """Devolve a config escolhida na VALIDAÇÃO (média dos folds) p/ cada modelo residual.
    Usado pelo 03 para congelar os hiperparâmetros antes de abrir o teste."""
    folds = mv.folds_config(cfg)
    preps = [prep_fold(df, fo, h, src) for fo in folds]
    escolhidos = {}
    for nome, grade in GRADE_RES.items():
        melhor = {"mae": np.inf, "config": None}
        for params in grade:
            maes = []
            for p in preps:
                mod = _criar_res(nome, params); mod.fit(p["Xtr"], p["r_tr"])
                pred = p["b3_va"] + mod.predict(p["Xva"])
                maes.append(mean_absolute_error(p["real_va"], pred))
            mm = float(np.mean(maes))
            if mm < melhor["mae"]:
                melhor = {"mae": round(mm, 4), "config": params}
        escolhidos[nome] = melhor
    return escolhidos


def testar_residual(df, cfg, h, config_xgb, src="era5") -> pd.DataFrame:
    """Treina o XGBRes residual com a config CONGELADA no treino purgado (< teste_inicio)
    e prevê o TESTE (>= teste_inicio). Reaproveita prep_fold com um 'fold' = bloco de teste.
    Retorna oof de teste com data, real, vol_t, B3, XGBRes."""
    fold_teste = {"nome": "TESTE", "ini": cfg["split"]["teste_inicio"],
                  "fim": df["data"].max().strftime("%Y-%m-%d")}
    p = prep_fold(df, fold_teste, h, src)
    mod = _criar_res("XGBRes", config_xgb); mod.fit(p["Xtr"], p["r_tr"])
    pred_vol = p["b3_va"] + mod.predict(p["Xva"])
    return pd.DataFrame({"data": p["datas_va"], "real": p["real_va"],
                         "vol_t": p["vol_t_va"], "B3": p["b3_va"], "XGBRes": pred_vol})


# ============================================================ avaliação condicional
def _skill_ci(datas, err_m, err_b3, rng, n_boot=600, bloco=90):
    ordem = np.argsort(datas)
    em, eb = np.asarray(err_m)[ordem], np.asarray(err_b3)[ordem]
    N = len(em)
    if N < 10 or eb.mean() <= 0:
        base = (1 - em.mean() / eb.mean()) if (N and eb.mean() > 0) else np.nan
        return base, np.nan, np.nan, N
    base = 1 - em.mean() / eb.mean()
    pool = np.arange(0, max(1, N - bloco + 1))
    sk = []
    for _ in range(n_boot):
        idx = []
        while len(idx) < N:
            s = int(rng.choice(pool)); idx.extend(range(s, min(s + bloco, N)))
        idx = np.array(idx[:N])
        bb = eb[idx].mean()
        sk.append(1 - em[idx].mean() / bb if bb > 0 else np.nan)
    lo, hi = np.nanpercentile(sk, [2.5, 97.5])
    return round(base, 4), round(float(lo), 4), round(float(hi), 4), N


def skill_condicional(oof_df, episodios40, modelos, h, seed=0):
    """MAE e skill vs B3 por subconjunto (geral / seca / pré-episódio), com IC95%
    por bootstrap em blocos de 90 dias."""
    rng = np.random.default_rng(seed)
    datas = pd.to_datetime(oof_df["data"])
    real = oof_df["real"].values
    err = {"B3": np.abs(real - oof_df["B3"].values)}
    for m in modelos:
        err[m] = np.abs(real - oof_df[m].values)

    # máscara pré-episódio (limiar 40): [ini-180, ini+30]
    pre = np.zeros(len(oof_df), bool)
    for _, ep in episodios40.iterrows():
        ini = pd.Timestamp(ep["inicio"])
        pre |= ((datas >= ini - pd.Timedelta(days=180)) & (datas <= ini + pd.Timedelta(days=30))).values
    seca = oof_df["spi12"].values < -1.0

    subconjuntos = {"geral": np.ones(len(oof_df), bool), "seca": seca,
                    "nao_seca": ~seca, "pre_episodio": pre}
    linhas = []
    for sub, mask in subconjuntos.items():
        if mask.sum() == 0:
            continue
        d_sub = datas[mask].values.astype("datetime64[D]").astype(float)
        for m in modelos:
            base, lo, hi, N = _skill_ci(d_sub, err[m][mask], err["B3"][mask], rng)
            linhas.append({
                "subconjunto": sub, "horizonte": h, "modelo": m, "n": int(N),
                "MAE": round(float(err[m][mask].mean()), 3),
                "MAE_B3": round(float(err["B3"][mask].mean()), 3),
                "skill_B3": base, "ic95_lo": lo, "ic95_hi": hi,
            })
    return linhas


# ============================================================ antecedência (lado a lado)
def _threshold_vol(oof_df, modelo, crise_label) -> float:
    """Escolhe o limiar sobre o volume PREVISTO que maximiza F1 na validação
    (alerta = previsão de volume abaixo do limiar)."""
    y = crise_label.astype(int).values
    if y.sum() == 0:
        return np.nan
    pred = oof_df[modelo].values
    melhor = (np.nan, -1)
    for th in np.arange(20, 60, 1.0):
        f1 = f1_score(y, (pred < th).astype(int), zero_division=0)
        if f1 > melhor[1]:
            melhor = (float(th), f1)
    return melhor[0]


def antecedencia_lado_a_lado(oof_df, base_diaria, episodios, modelos, h, limiar,
                             dias_olhar=120):
    """Para cada modelo, usa a previsão OOF de volume; alerta quando prev<threshold
    (threshold por F1 na validação). Mede antecedência e FALSO-ALARME por EPISÓDIO/ano."""
    datas = pd.to_datetime(oof_df["data"])
    crise_label = pd.Series(oof_df["real"].values < limiar, index=datas.values)
    cob_ini, cob_fim = datas.min(), datas.max()

    # série diária real de volume (p/ checar se o alerta é seguido de crise)
    vol = base_diaria.set_index("data")["vol_util_pct_sistema"].sort_index()
    crise_real = vol < limiar

    eps = episodios.copy()
    eps["ini"] = pd.to_datetime(eps["inicio"])
    eps = eps[(eps["ini"] >= cob_ini) & (eps["ini"] <= cob_fim)]

    linhas = []
    for modelo in modelos:
        th = _threshold_vol(oof_df, modelo, crise_label)
        alerta = pd.Series((oof_df[modelo].values < th), index=datas.values).sort_index()
        # falso-alarme por EPISÓDIO: run contínuo de alerta não seguido de crise em h+30 d
        fa_runs = _runs(alerta)
        falsos = 0
        for a, b in fa_runs:
            fim_janela = b + pd.Timedelta(days=h + 30)
            jc = crise_real[(crise_real.index >= a) & (crise_real.index <= fim_janela)]
            if not bool(jc.any()):
                falsos += 1
        anos = max(1, (cob_fim - cob_ini).days / 365.25)
        fa_por_ano = round(falsos / anos, 2)

        for _, ep in eps.iterrows():
            ini = ep["ini"]
            jan = alerta[(alerta.index >= ini - pd.Timedelta(days=dias_olhar)) &
                         (alerta.index < ini)]
            disp = jan[jan]
            if len(disp):
                prim = disp.index.min(); ant = int((ini - prim).days); ps = prim.strftime("%Y-%m-%d")
            else:
                ant, ps = np.nan, "sem alerta"
            linhas.append({
                "episodio_inicio": ini.strftime("%Y-%m-%d"), "limiar": limiar,
                "horizonte": h, "modelo": modelo, "threshold_vol": round(th, 1) if np.isfinite(th) else np.nan,
                "primeiro_alerta": ps, "antecedencia_dias": ant,
                "falso_alarme_episodios_ano": fa_por_ano,
            })
    return linhas


def _alertas_e_falsos(oof_df, base_diaria, modelo, crise_label, h, limiar):
    """Devolve (serie_alerta, threshold, falsos_abs, falsos_ano, cob_ini, cob_fim)."""
    datas = pd.to_datetime(oof_df["data"])
    th = _threshold_vol(oof_df, modelo, crise_label)
    alerta = pd.Series((oof_df[modelo].values < th), index=datas.values).sort_index()
    vol = base_diaria.set_index("data")["vol_util_pct_sistema"].sort_index()
    crise_real = vol < limiar
    falsos = 0
    for a, b in _runs(alerta):
        jc = crise_real[(crise_real.index >= a) & (crise_real.index <= b + pd.Timedelta(days=h + 30))]
        if not bool(jc.any()):
            falsos += 1
    cob_ini, cob_fim = datas.min(), datas.max()
    anos = max(1.0, (cob_fim - cob_ini).days / 365.25)
    return alerta, th, falsos, round(falsos / anos, 3), cob_ini, cob_fim


def _antecedencia_episodios(alerta, eps_ini, dias_olhar=120):
    """Antecedência (dias) por episódio; 0 se não houve alerta na janela (p/ o pareado)."""
    out = {}
    for ini in eps_ini:
        jan = alerta[(alerta.index >= ini - pd.Timedelta(days=dias_olhar)) & (alerta.index < ini)]
        disp = jan[jan]
        out[ini] = int((ini - disp.index.min()).days) if len(disp) else 0
    return out


def antecedencia_pareada(oof_df, base_diaria, episodios, h, limiar,
                         modelo_a="XGBRes", modelo_b="B3", n_boot=2000, seed=0):
    """Teste PAREADO da antecedência (modelo_a − modelo_b) por episódio:
    mediana da diferença, IC95% bootstrap sobre episódios e Wilcoxon pareado.
    Retorna (linhas_por_episodio, resumo)."""
    from scipy.stats import wilcoxon
    datas = pd.to_datetime(oof_df["data"])
    crise_label = pd.Series(oof_df["real"].values < limiar, index=datas.values)
    eps = episodios.copy(); eps["ini"] = pd.to_datetime(eps["inicio"])
    eps = eps[(eps["ini"] >= datas.min()) & (eps["ini"] <= datas.max())]
    eps_ini = list(eps["ini"])

    al_a, th_a, fa_a_abs, fa_a_ano, *_ = _alertas_e_falsos(oof_df, base_diaria, modelo_a,
                                                           crise_label, h, limiar)
    al_b, th_b, fa_b_abs, fa_b_ano, *_ = _alertas_e_falsos(oof_df, base_diaria, modelo_b,
                                                           crise_label, h, limiar)
    ant_a = _antecedencia_episodios(al_a, eps_ini)
    ant_b = _antecedencia_episodios(al_b, eps_ini)

    por_ep, difs = [], []
    for ini in eps_ini:
        da, db = ant_a[ini], ant_b[ini]
        difs.append(da - db)
        por_ep.append({"episodio_inicio": ini.strftime("%Y-%m-%d"), "limiar": limiar,
                       "horizonte": h, f"ant_{modelo_a}": da, f"ant_{modelo_b}": db,
                       "dif": da - db})
    difs = np.array(difs, float)

    rng = np.random.default_rng(seed)
    if len(difs) >= 2:
        boot = [np.median(rng.choice(difs, len(difs), replace=True)) for _ in range(n_boot)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
    else:
        lo = hi = np.nan
    try:
        w_stat, w_p = wilcoxon(difs) if np.any(difs != 0) else (np.nan, 1.0)
    except Exception:  # noqa: BLE001
        w_stat, w_p = np.nan, np.nan

    resumo = {
        "limiar": limiar, "horizonte": h, "n_episodios": len(difs),
        "mediana_dif": float(np.median(difs)) if len(difs) else np.nan,
        "ic95_lo": round(float(lo), 2) if np.isfinite(lo) else np.nan,
        "ic95_hi": round(float(hi), 2) if np.isfinite(hi) else np.nan,
        "wilcoxon_stat": round(float(w_stat), 3) if np.isfinite(w_stat) else np.nan,
        "wilcoxon_p": round(float(w_p), 4) if np.isfinite(w_p) else np.nan,
        f"falso_alarme_ano_{modelo_a}": fa_a_ano, f"falso_alarme_abs_{modelo_a}": fa_a_abs,
        f"falso_alarme_ano_{modelo_b}": fa_b_ano, f"falso_alarme_abs_{modelo_b}": fa_b_abs,
        f"threshold_{modelo_a}": round(th_a, 1) if np.isfinite(th_a) else np.nan,
        f"threshold_{modelo_b}": round(th_b, 1) if np.isfinite(th_b) else np.nan,
    }
    return por_ep, resumo


def _runs(booleana: pd.Series):
    """Retorna lista de (inicio, fim) dos trechos contínuos True (datas no índice)."""
    v = booleana.values; idx = booleana.index
    runs, i, n = [], 0, len(v)
    while i < n:
        if v[i]:
            j = i
            while j + 1 < n and v[j + 1]:
                j += 1
            runs.append((idx[i], idx[j])); i = j + 1
        else:
            i += 1
    return runs
