# -*- coding: utf-8 -*-
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import analise_exp3 as ax
import transformadores as tr

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data" / "processed_v2" / "dataset_diario.parquet"
CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"


# -------------------------- residual + B3 reconstrói vol(t+h)
@pytest.mark.skipif(not DATASET.exists(), reason="dataset v2 ainda não gerado")
def test_residual_mais_b3_reconstroi_alvo():
    df = pd.read_parquet(DATASET); df["data"] = pd.to_datetime(df["data"])
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    fold = cfg["split"]["folds"][2]          # 2013-2017 (contém a grande crise)
    for h in (30, 90):
        p = ax.prep_fold(df, fold, h, src="era5")
        # no treino: B3 + resíduo == volume real em t+h (exato)
        assert np.allclose(p["b3_tr"] + p["r_tr"], p["real_tr"], atol=1e-9)
        # definição do resíduo coerente
        assert np.allclose(p["r_tr"], p["real_tr"] - p["b3_tr"], atol=1e-12)


# -------------------------- PR-AUC de entrada (IC) é determinística e bem formada
def test_prauc_entrada_ic_horizonte_deterministico():
    n = 600
    datas = pd.date_range("2003-01-01", periods=n, freq="D")
    gen = np.random.default_rng(7)
    vol_t = gen.uniform(35, 70, n)            # vol(t): parte acima de 40 (recorte de entrada)
    real = vol_t - gen.uniform(-5, 25, n)     # vol(t+h): às vezes abaixo de 40 (entra em crise)
    oof = pd.DataFrame({"data": datas, "vol_t": vol_t, "real": real,
                        "B3": real + gen.normal(0, 3, n),
                        "XGBRes": real + gen.normal(0, 2, n)})
    ent = vol_t > 40
    y = real[ent] < 40
    assert y.any() and (~y).any()             # há as duas classes no recorte de entrada
    r1 = ax.prauc_entrada_ic_horizonte(oof, 30, np.random.default_rng(0), n_boot=200)
    r2 = ax.prauc_entrada_ic_horizonte(oof, 30, np.random.default_rng(0), n_boot=200)
    assert r1 == r2                           # mesmo rng + mesmo oof -> saída idêntica
    assert [r["modelo"] for r in r1] == ["B3", "XGBRes"]
    for row in r1:
        assert 0.0 < row["prevalencia"] <= 1.0
        assert row["ic95_lo"] <= row["ic95_hi"]
        assert row["horizonte"] == 30


# -------------------------- anomalias usam só a climatologia do treino
def test_anomalias_usam_so_climatologia_do_treino():
    datas = pd.date_range("1990-01-01", periods=4000, freq="D")
    rng = np.random.default_rng(0)
    val = rng.gamma(2.0, 20.0, len(datas))
    df = pd.DataFrame({"data": datas, "acc": val})
    treino = (df["data"].dt.year < 2000).values

    cl1 = tr.ClimatologiaSazonal().fit(df.loc[treino, "data"], df.loc[treino, "acc"].values)
    an1 = tr.anomalias_sazonais(df, treino, ["acc"])

    # altera dados de VALIDAÇÃO (>=2000); a climatologia (fit no treino) não pode mudar
    df2 = df.copy()
    df2.loc[~treino, "acc"] *= 9.0
    cl2 = tr.ClimatologiaSazonal().fit(df2.loc[treino, "data"], df2.loc[treino, "acc"].values)
    an2 = tr.anomalias_sazonais(df2, treino, ["acc"])

    assert cl1.clim_ == cl2.clim_, "a climatologia mudou ao alterar a validação"
    # a anomalia das linhas de TREINO não muda
    assert np.allclose(an1.loc[treino, "anom_acc"], an2.loc[treino, "anom_acc"])
    # anomalia = valor − climatologia[doy]
    esperado = df.loc[treino, "acc"].values - np.array(
        [cl1.clim_[min(d, 365)] for d in pd.to_datetime(df.loc[treino, "data"]).dt.dayofyear])
    assert np.allclose(an1.loc[treino, "anom_acc"].values, esperado)
