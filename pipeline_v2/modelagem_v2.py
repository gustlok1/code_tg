# -*- coding: utf-8 -*-
"""
modelagem_v2.py — Núcleo (importável) da validação temporal do pipeline v2.

Contém: split com janela crescente + PURGA por horizonte, construção dos conjuntos
de features (com SPI/SPEI ajustados SÓ no treino de cada fold), baselines B1/B2/B3,
modelos (Ridge, LogReg, RandomForest, XGBoost), métricas (regressão e classificação)
e a análise de antecedência por episódio.

O período de TESTE (2023+) nunca é tocado aqui; quem decide abri-lo é o 02_treino_v2.py.
Separado de 02_* porque um módulo com nome começando em dígito não é importável.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             mean_absolute_error, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

import transformadores as tr

# ---------------------------------------------------------------- features
CLIMA_ERA5 = [
    "era5_precip_acc7", "era5_precip_acc30", "era5_precip_acc90", "era5_precip_acc180",
    "era5_precip_acc365", "et0_acc7", "et0_acc30", "et0_acc90", "et0_acc180", "et0_acc365",
    "era5_bal_acc7", "era5_bal_acc30", "era5_bal_acc90", "era5_bal_acc180", "era5_bal_acc365",
    "era5_tmean_ma7", "era5_tmean_ma30", "doy_sin", "doy_cos",
]
SPI_COLS = ["spi_3", "spi_6", "spi_12", "spei_3", "spei_6", "spei_12"]
ESTADO = ["vol_pct", "vol_var7", "vol_var30", "aflu_ma7", "aflu_ma30", "aflu_ma90",
          "deflu_ma30", "pos_2017"]

CONJUNTOS = {
    "CLIMA": CLIMA_ERA5 + SPI_COLS,
    "CLIMA_ESTADO": CLIMA_ERA5 + SPI_COLS + ESTADO,
}

# ---------------------------------------------------------------- grades (pequenas)
GRADE_REG = {
    "Ridge": ("linear", [{"alpha": 1.0}, {"alpha": 10.0}]),
    "RandomForestReg": ("arvore", [{"n_estimators": 200, "max_depth": None},
                                   {"n_estimators": 200, "max_depth": 10}]),
    "XGBReg": ("arvore", [{"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05},
                          {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05}]),
}
GRADE_CLF = {
    "LogReg": ("linear", [{"C": 0.1}, {"C": 1.0}]),
    "RandomForestClf": ("arvore", [{"n_estimators": 300, "max_depth": None},
                                   {"n_estimators": 300, "max_depth": 8}]),
    "XGBClf": ("arvore", [{"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05},
                          {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.05}]),
}
SEED = 42


# ============================================================ split + purga
def mascaras_fold(datas: pd.Series, ini, fim, h: int):
    """Train = tudo ANTES de `ini` cujo alvo (t+h) ainda cai antes de `ini` (purga).
    Val = [ini, fim]."""
    datas = pd.to_datetime(datas)
    ini, fim = pd.Timestamp(ini), pd.Timestamp(fim)
    val = (datas >= ini) & (datas <= fim)
    train = (datas + pd.to_timedelta(h, "D")) < ini      # já implica datas < ini
    return train.values, val.values


def folds_config(cfg):
    return cfg["split"]["folds"]


# ============================================================ SPI/SPEI por fold
def spi_spei_do_fold(df: pd.DataFrame, train_mask: np.ndarray,
                     escalas=(3, 6, 12)) -> pd.DataFrame:
    """Ajusta SPI/SPEI SÓ nas linhas de treino e transforma a série inteira.
    Retorna DataFrame (índice de df) com spi_3/6/12 e spei_3/6/12."""
    return tr.aplicar_indicadores(
        df, pd.Series(train_mask, index=df.index),
        col_data="data", col_precip="era5_precip_mm", col_et0="era5_et0_mm",
        escalas=escalas)


# ============================================================ baselines
def baseline_B1(df: pd.DataFrame) -> np.ndarray:
    """Persistência: previsão = volume em t."""
    return df["vol_pct"].values.astype(float)


def _doy(datas: pd.Series) -> np.ndarray:
    d = pd.to_datetime(datas).dt.dayofyear.values
    return np.minimum(d, 365)


def ajustar_B2(df_tr: pd.DataFrame) -> dict:
    """Climatologia do volume por dia do ano (ajuste no treino)."""
    doy = _doy(df_tr["data"])
    s = pd.Series(df_tr["vol_pct"].values, index=doy)
    return s.groupby(level=0).mean().to_dict()


def prever_B2(df: pd.DataFrame, h: int, clim: dict) -> np.ndarray:
    """B2: volume climatológico no dia do ano de t+h."""
    doy_alvo = _doy(pd.to_datetime(df["data"]) + pd.to_timedelta(h, "D"))
    media_geral = np.nanmean(list(clim.values())) if clim else np.nan
    return np.array([clim.get(d, media_geral) for d in doy_alvo], dtype=float)


def ajustar_B3(df_tr: pd.DataFrame, h: int) -> dict:
    """Variação sazonal média em h dias por dia do ano (ajuste no treino)."""
    doy = _doy(df_tr["data"])
    var = df_tr[f"vol_var_t{h}"].values.astype(float)
    s = pd.Series(var, index=doy).dropna()
    return s.groupby(level=0).mean().to_dict()


def prever_B3(df: pd.DataFrame, h: int, delta: dict) -> np.ndarray:
    """B3 (baseline difícil): volume em t + variação sazonal média em h dias."""
    doy = _doy(df["data"])
    media_geral = np.nanmean(list(delta.values())) if delta else 0.0
    d = np.array([delta.get(x, media_geral) for x in doy], dtype=float)
    return df["vol_pct"].values.astype(float) + d


# ============================================================ modelos
def _criar_reg(nome, params):
    if nome == "Ridge":
        return Pipeline([("sc", StandardScaler()), ("m", Ridge(**params))])
    if nome == "RandomForestReg":
        return RandomForestRegressor(random_state=SEED, n_jobs=-1, **params)
    if nome == "XGBReg":
        return XGBRegressor(random_state=SEED, n_jobs=-1, objective="reg:squarederror", **params)
    raise ValueError(nome)


def _criar_clf(nome, params, scale_pos_weight=1.0):
    if nome == "LogReg":
        return Pipeline([("sc", StandardScaler()),
                         ("m", LogisticRegression(max_iter=2000, class_weight="balanced", **params))])
    if nome == "RandomForestClf":
        return RandomForestClassifier(random_state=SEED, n_jobs=-1,
                                      class_weight="balanced", **params)
    if nome == "XGBClf":
        return XGBClassifier(random_state=SEED, n_jobs=-1, eval_metric="logloss",
                             scale_pos_weight=scale_pos_weight, **params)
    raise ValueError(nome)


# ============================================================ métricas
def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def metricas_reg(y, pred, b1, b3) -> dict:
    mae = mean_absolute_error(y, pred)
    mae_b1 = mean_absolute_error(y, b1)
    mae_b3 = mean_absolute_error(y, b3)
    return {
        "MAE": round(float(mae), 3), "RMSE": round(_rmse(y, pred), 3),
        "skill_B1": round(1 - mae / mae_b1, 4) if mae_b1 > 0 else np.nan,
        "skill_B3": round(1 - mae / mae_b3, 4) if mae_b3 > 0 else np.nan,
    }


def metricas_clf(y, score, prob=None) -> dict:
    y = np.asarray(y).astype(int)
    n, pos = len(y), int(y.sum())
    base = {"n": n, "prevalencia": round(pos / n, 4) if n else np.nan}
    if pos == 0 or pos == n:
        base.update({"PR_AUC": "sem positivos", "ROC_AUC": "sem positivos",
                     "Brier": np.nan})
        return base
    base["PR_AUC"] = round(float(average_precision_score(y, score)), 4)
    base["ROC_AUC"] = round(float(roc_auc_score(y, score)), 4)
    base["Brier"] = round(float(brier_score_loss(y, prob)), 4) if prob is not None else np.nan
    return base


def escolher_threshold(y, prob) -> float:
    """Threshold que maximiza F1 (escolhido na validação)."""
    y = np.asarray(y).astype(int)
    if y.sum() == 0:
        return 0.5
    melhores = (0.5, -1)
    for t in np.arange(0.05, 0.96, 0.05):
        f1 = f1_score(y, (prob >= t).astype(int), zero_division=0)
        if f1 > melhores[1]:
            melhores = (float(t), f1)
    return melhores[0]


# ============================================================ validação (regressão)
def validar_regressao(df, cfg, conjunto, h, spi_por_fold) -> list:
    """Retorna linhas de métricas (por fold) p/ baselines e cada modelo de regressão,
    escolhendo hiperparâmetros pela média dos folds."""
    folds = folds_config(cfg)
    feats = CONJUNTOS[conjunto]
    usa_delta = conjunto == "CLIMA_ESTADO"
    alvo = f"vol_var_t{h}" if usa_delta else f"vol_pct_t{h}"

    feats_df = [c for c in feats if c not in SPI_COLS]
    # pré-computa, por fold, os dados prontos
    dados = []
    for k, fo in enumerate(folds):
        trm, vam = mascaras_fold(df["data"], fo["ini"], fo["fim"], h)
        Xf = df[feats_df].join(spi_por_fold[k])          # junta as features de clima/estado + SPI
        ok_cols = Xf.notna().all(axis=1).values
        tr_ok = trm & ok_cols & df[alvo].notna().values
        va_ok = vam & ok_cols & df[f"vol_pct_t{h}"].notna().values
        d_tr, d_va = df[tr_ok], df[va_ok]
        Xtr = Xf[tr_ok]; Xva = Xf[va_ok]
        ytr = df.loc[tr_ok, alvo].values
        yva_vol = df.loc[va_ok, f"vol_pct_t{h}"].values
        volt_va = df.loc[va_ok, "vol_pct"].values
        # baselines (ajustados no treino do fold)
        b1 = baseline_B1(d_va)
        b2 = prever_B2(d_va, h, ajustar_B2(d_tr))
        b3 = prever_B3(d_va, h, ajustar_B3(d_tr, h))
        dados.append(dict(Xtr=Xtr, ytr=ytr, Xva=Xva, yva_vol=yva_vol, volt_va=volt_va,
                          datas_va=d_va["data"].values, b1=b1, b2=b2, b3=b3, fold=fo["nome"]))

    linhas, oof_reg = [], {}

    def _guardar_oof(nome, preds):
        oof_reg[nome] = dict(
            data=np.concatenate([d["datas_va"] for d in dados]),
            pred=np.concatenate([np.asarray(p) for p in preds]),
            real=np.concatenate([d["yva_vol"] for d in dados]))

    # --- baselines como "modelos" ---
    for nome_b, chave in [("B1_persistencia", "b1"), ("B2_climatologia", "b2"),
                          ("B3_persist_sazonal", "b3")]:
        preds = [d[chave] for d in dados]
        for d, pred in zip(dados, preds):
            m = metricas_reg(d["yva_vol"], pred, d["b1"], d["b3"])
            linhas.append(_linha(d["fold"], nome_b, conjunto, h, None, "regressao",
                                 "—", m, config="—"))
        _guardar_oof(nome_b, preds)

    # --- modelos (seleção de hiperparâmetro pela média dos folds) ---
    for nome, (tipo, grade) in GRADE_REG.items():
        melhor = _selecionar_reg(nome, grade, dados, usa_delta)
        for d, pred in zip(dados, melhor["preds"]):
            m = metricas_reg(d["yva_vol"], pred, d["b1"], d["b3"])
            linhas.append(_linha(d["fold"], nome, conjunto, h, None, "regressao",
                                 "—", m, config=str(melhor["config"])))
        _guardar_oof(nome, melhor["preds"])
    return linhas, oof_reg


def _selecionar_reg(nome, grade, dados, usa_delta):
    melhor = {"mae": np.inf, "config": None, "preds": None}
    for params in grade:
        preds, maes = [], []
        for d in dados:
            mod = _criar_reg(nome, params)
            mod.fit(d["Xtr"], d["ytr"])
            p = mod.predict(d["Xva"])
            pred_vol = (d["volt_va"] + p) if usa_delta else p   # árvore não extrapola nível
            preds.append(pred_vol)
            maes.append(mean_absolute_error(d["yva_vol"], pred_vol))
        mm = float(np.mean(maes))
        if mm < melhor["mae"]:
            melhor = {"mae": mm, "config": params, "preds": preds}
    return melhor


# ============================================================ validação (classificação)
def validar_classificacao(df, cfg, conjunto, h, limiar, spi_por_fold) -> tuple:
    """Retorna (linhas_metricas, oof) p/ baselines e modelos de classificação, nos dois
    recortes (todos; só entrada = vol(t) acima do limiar). oof = dict p/ análise de episódios."""
    folds = folds_config(cfg)
    feats = CONJUNTOS[conjunto]
    alvo = f"crise_{h}" if limiar == 30 else f"crise{limiar}_{h}"

    feats_df = [c for c in feats if c not in SPI_COLS]
    dados = []
    for k, fo in enumerate(folds):
        trm, vam = mascaras_fold(df["data"], fo["ini"], fo["fim"], h)
        Xf = df[feats_df].join(spi_por_fold[k])
        ok_cols = Xf.notna().all(axis=1).values
        tr_ok = trm & ok_cols & df[alvo].notna().values
        va_ok = vam & ok_cols & df[alvo].notna().values
        d_tr, d_va = df[tr_ok], df[va_ok]
        ytr = df.loc[tr_ok, alvo].astype(int).values
        yva = df.loc[va_ok, alvo].astype(int).values
        volt_va = df.loc[va_ok, "vol_pct"].values
        spi6_va = Xf.loc[va_ok, "spi_6"].values
        # baselines de regressão -> vira crise se pred_vol < limiar (score = -pred_vol)
        b1 = baseline_B1(d_va)
        b3 = prever_B3(d_va, h, ajustar_B3(d_tr, h))
        dados.append(dict(Xtr=Xf[tr_ok], ytr=ytr, Xva=Xf[va_ok], yva=yva, volt_va=volt_va,
                          spi6_va=spi6_va, datas_va=d_va["data"].values,
                          b1=b1, b3=b3, fold=fo["nome"]))

    linhas, oof = [], {}

    def _emit(nome, scores_por_fold, probs_por_fold, config="—"):
        oof_prob, oof_y, oof_vol, oof_dat = [], [], [], []
        for d, score, prob in zip(dados, scores_por_fold, probs_por_fold):
            for recorte in ("todos", "entrada"):
                if recorte == "todos":
                    sel = np.ones(len(d["yva"]), bool)
                else:
                    sel = d["volt_va"] > limiar          # só dias ainda fora da crise
                if sel.sum() == 0:
                    continue
                m = metricas_clf(d["yva"][sel], np.asarray(score)[sel],
                                 None if prob is None else np.asarray(prob)[sel])
                linhas.append(_linha(d["fold"], nome, conjunto, h, limiar, "classificacao",
                                     recorte, m, config=str(config)))
            if prob is not None:
                oof_prob.append(np.asarray(prob)); oof_y.append(d["yva"])
                oof_vol.append(d["volt_va"]); oof_dat.append(d["datas_va"])
        if prob is not None and oof_prob:
            oof[nome] = dict(prob=np.concatenate(oof_prob), y=np.concatenate(oof_y),
                             vol=np.concatenate(oof_vol),
                             data=np.concatenate(oof_dat))
    # baselines
    _emit("B1_persistencia", [-d["b1"] for d in dados], [None] * len(dados))
    _emit("B3_persist_sazonal", [-d["b3"] for d in dados], [None] * len(dados))
    _emit("regra_SPI6<-1", [-d["spi6_va"] for d in dados], [None] * len(dados))

    # modelos
    for nome, (tipo, grade) in GRADE_CLF.items():
        melhor = _selecionar_clf(nome, grade, dados)
        _emit(nome, melhor["probs"], melhor["probs"], config=melhor["config"])
    return linhas, oof


def _selecionar_clf(nome, grade, dados):
    melhor = {"ap": -np.inf, "config": None, "probs": None}
    for params in grade:
        probs, aps = [], []
        for d in dados:
            spw = (d["ytr"] == 0).sum() / max((d["ytr"] == 1).sum(), 1)
            mod = _criar_clf(nome, params, scale_pos_weight=spw)
            mod.fit(d["Xtr"], d["ytr"])
            p = mod.predict_proba(d["Xva"])[:, 1]
            probs.append(p)
            if d["yva"].sum() > 0:
                aps.append(average_precision_score(d["yva"], p))
        mm = float(np.mean(aps)) if aps else -np.inf
        if mm > melhor["ap"]:
            melhor = {"ap": mm, "config": params, "probs": probs}
    return melhor


# ============================================================ linha de métrica
def _linha(fold, modelo, conjunto, h, limiar, tarefa, recorte, met, config="—") -> dict:
    base = {"fold": fold, "modelo": modelo, "conjunto": conjunto, "horizonte": h,
            "limiar": limiar if limiar is not None else "—", "tarefa": tarefa,
            "recorte": recorte, "config": config,
            "MAE": "", "RMSE": "", "skill_B1": "", "skill_B3": "",
            "n": "", "prevalencia": "", "PR_AUC": "", "ROC_AUC": "", "Brier": ""}
    base.update({k: v for k, v in met.items()})
    return base


# ============================================================ TESTE (dormente)
def _config_da_validacao(val_linhas, tarefa, conjunto, h, modelo, limiar=None):
    import ast
    for r in val_linhas:
        if (r["tarefa"] == tarefa and r["conjunto"] == conjunto and r["horizonte"] == h
                and r["modelo"] == modelo
                and (limiar is None or r["limiar"] == limiar)):
            if r["config"] not in ("—", "", None):
                try:
                    return ast.literal_eval(r["config"])
                except Exception:  # noqa: BLE001
                    return {}
    return {}


def avaliar_teste(df, cfg, conjunto, h, val_linhas) -> list:
    """Avalia no TESTE (>= teste_inicio) com treino purgado (t+h < teste_inicio) e as
    configs escolhidas na validação. SÓ chamado sob --abrir-teste."""
    teste_inicio = cfg["split"]["teste_inicio"]
    trm, tem = mascaras_fold(df["data"], teste_inicio, df["data"].max(), h)
    spi = spi_spei_do_fold(df, trm)
    feats = CONJUNTOS[conjunto]
    Xf = df[[c for c in feats if c not in SPI_COLS]].join(spi)
    ok = Xf.notna().all(axis=1).values
    usa_delta = conjunto == "CLIMA_ESTADO"
    alvo = f"vol_var_t{h}" if usa_delta else f"vol_pct_t{h}"
    tr_ok = trm & ok & df[alvo].notna().values
    te_ok = tem & ok & df[f"vol_pct_t{h}"].notna().values
    d_tr, d_te = df[tr_ok], df[te_ok]
    yte_vol = df.loc[te_ok, f"vol_pct_t{h}"].values
    volt_te = df.loc[te_ok, "vol_pct"].values
    b1 = baseline_B1(d_te); b3 = prever_B3(d_te, h, ajustar_B3(d_tr, h))
    linhas = []

    # regressão
    for nome_b, pred in [("B1_persistencia", b1),
                         ("B2_climatologia", prever_B2(d_te, h, ajustar_B2(d_tr))),
                         ("B3_persist_sazonal", b3)]:
        linhas.append(_linha("TESTE", nome_b, conjunto, h, None, "regressao", "—",
                             metricas_reg(yte_vol, pred, b1, b3)))
    for nome in GRADE_REG:
        params = _config_da_validacao(val_linhas, "regressao", conjunto, h, nome)
        mod = _criar_reg(nome, params); mod.fit(Xf[tr_ok], df.loc[tr_ok, alvo].values)
        p = mod.predict(Xf[te_ok]); pred_vol = (volt_te + p) if usa_delta else p
        linhas.append(_linha("TESTE", nome, conjunto, h, None, "regressao", "—",
                             metricas_reg(yte_vol, pred_vol, b1, b3), config=str(params)))

    # classificação (limiares 40 e 30, recortes todos/entrada)
    for limiar in (40, 30):
        alvo_c = f"crise_{h}" if limiar == 30 else f"crise{limiar}_{h}"
        trc = trm & ok & df[alvo_c].notna().values
        tec = tem & ok & df[alvo_c].notna().values
        yte = df.loc[tec, alvo_c].astype(int).values
        volt = df.loc[tec, "vol_pct"].values
        spi6 = Xf.loc[tec, "spi_6"].values
        pred_b1 = baseline_B1(df[tec]); pred_b3 = prever_B3(df[tec], h, ajustar_B3(df[trc], h))
        scores_base = {"B1_persistencia": -pred_b1, "B3_persist_sazonal": -pred_b3,
                       "regra_SPI6<-1": -spi6}
        for nome, score in scores_base.items():
            for recorte in ("todos", "entrada"):
                sel = np.ones(len(yte), bool) if recorte == "todos" else (volt > limiar)
                if sel.sum() == 0:
                    continue
                linhas.append(_linha("TESTE", nome, conjunto, h, limiar, "classificacao",
                                     recorte, metricas_clf(yte[sel], score[sel])))
        for nome in GRADE_CLF:
            params = _config_da_validacao(val_linhas, "classificacao", conjunto, h, nome, limiar)
            ytr = df.loc[trc, alvo_c].astype(int).values
            spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
            mod = _criar_clf(nome, params, scale_pos_weight=spw)
            mod.fit(Xf[trc], ytr)
            prob = mod.predict_proba(Xf[tec])[:, 1]
            for recorte in ("todos", "entrada"):
                sel = np.ones(len(yte), bool) if recorte == "todos" else (volt > limiar)
                if sel.sum() == 0:
                    continue
                linhas.append(_linha("TESTE", nome, conjunto, h, limiar, "classificacao",
                                     recorte, metricas_clf(yte[sel], prob[sel], prob[sel]),
                                     config=str(params)))
    return linhas


# ============================================================ episódios (OOF)
def antecedencia_episodios(oof: dict, episodios: pd.DataFrame, cfg, conjunto, modelo,
                           h, limiar, dias_olhar=120) -> list:
    """Para o modelo escolhido, usa as previsões OOF de validação para medir:
    data do 1º alerta nos `dias_olhar` dias antes do início, antecedência, e
    alarmes falsos/ano fora dos episódios."""
    if modelo not in oof:
        return []
    o = oof[modelo]
    serie = pd.Series(o["prob"], index=pd.to_datetime(o["data"])).sort_index()
    serie = serie[~serie.index.duplicated()]
    yserie = pd.Series(o["y"], index=pd.to_datetime(o["data"])).sort_index()
    yserie = yserie[~yserie.index.duplicated()]
    thr = escolher_threshold(yserie.values, serie.values)
    alerta = serie >= thr

    cobertura_ini, cobertura_fim = serie.index.min(), serie.index.max()
    # episódios dentro da cobertura OOF (validação)
    eps = episodios.copy()
    eps["ini"] = pd.to_datetime(eps["inicio"]); eps["fim_dt"] = pd.to_datetime(eps["fim"])
    eps = eps[(eps["ini"] >= cobertura_ini) & (eps["ini"] <= cobertura_fim)]

    # alarmes falsos/ano: dias com alerta fora de qualquer episódio
    dentro = pd.Series(False, index=serie.index)
    for _, ep in eps.iterrows():
        dentro |= (serie.index >= ep["ini"]) & (serie.index <= ep["fim_dt"])
    fora = alerta & (~dentro)
    anos = fora.index.year
    fa_por_ano = round(float(fora.groupby(anos).sum().mean()), 2) if len(fora) else np.nan

    linhas = []
    for _, ep in eps.iterrows():
        ini = ep["ini"]
        jan = alerta[(alerta.index >= ini - pd.Timedelta(days=dias_olhar)) &
                     (alerta.index < ini)]
        primeiros = jan[jan]
        if len(primeiros):
            primeiro = primeiros.index.min()
            antecedencia = int((ini - primeiro).days)
            prim_str = primeiro.strftime("%Y-%m-%d")
        else:
            antecedencia, prim_str = np.nan, "sem alerta"
        linhas.append({
            "episodio_inicio": ini.strftime("%Y-%m-%d"), "episodio_fim": ep["fim"],
            "conjunto": conjunto, "modelo": modelo, "horizonte": h, "limiar": limiar,
            "threshold_val": round(thr, 2), "primeiro_alerta": prim_str,
            "antecedencia_dias": antecedencia, "alarmes_falsos_por_ano": fa_por_ano,
        })
    return linhas
