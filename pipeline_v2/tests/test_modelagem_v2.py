# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

import modelagem_v2 as mv
import transformadores as tr


# --------------------------------------------------------------- purga
def test_purga_nenhum_treino_invade_validacao():
    datas = pd.date_range("1990-01-01", periods=5000, freq="D")
    for h in (30, 60, 90):
        trm, vam = mv.mascaras_fold(pd.Series(datas), "2000-01-01", "2001-12-31", h)
        dtr = datas[trm]
        # todo t de treino tem t+h ainda antes do início da validação
        assert (dtr + pd.Timedelta(days=h)).max() < pd.Timestamp("2000-01-01"), \
            f"treino invade a validação em h={h}"
        # validação é exatamente o bloco
        assert datas[vam].min() == pd.Timestamp("2000-01-01")
        assert datas[vam].max() == pd.Timestamp("2001-12-31")


# ------------------------------------- SPI ajustado no fold não depende da validação
def test_spi_do_fold_invariante_a_validacao():
    rng = np.random.default_rng(0)
    idx = pd.date_range("1990-01-31", periods=30 * 12, freq="ME")
    mensal = pd.Series(rng.gamma(1.5, 5.0, len(idx)), index=idx)
    treino = mensal.index.year < 2010

    p1 = tr.SPI(3).fit(mensal[treino]).params_

    mensal2 = mensal.copy()
    mensal2[mensal2.index.year >= 2010] *= 7.0     # altera só a validação/futuro
    p2 = tr.SPI(3).fit(mensal2[treino]).params_

    assert p1 == p2, "os parâmetros do SPI mudaram ao alterar dados de validação"


# --------------------------------------------------------------- baselines
def test_B1_persistencia():
    df = pd.DataFrame({"vol_pct": [10.0, 20.0, 30.0]})
    assert list(mv.baseline_B1(df)) == [10.0, 20.0, 30.0]


def test_B2_climatologia():
    treino = pd.DataFrame({"data": pd.to_datetime(["2000-01-31", "2001-01-31"]),
                           "vol_pct": [40.0, 60.0]})           # doy 31 -> média 50
    clim = mv.ajustar_B2(treino)
    assert clim[31] == 50.0
    # t = 01/01; t+30 = 31/01 (doy 31) -> previsão 50
    pred = mv.prever_B2(pd.DataFrame({"data": pd.to_datetime(["2010-01-01"])}), 30, clim)
    assert pred[0] == 50.0


def test_B3_persistencia_mais_variacao_sazonal():
    treino = pd.DataFrame({"data": pd.to_datetime(["2000-01-01", "2001-01-01"]),
                           "vol_pct": [50.0, 60.0],
                           "vol_var_t30": [-10.0, -20.0]})     # doy 1 -> variação média -15
    delta = mv.ajustar_B3(treino, 30)
    assert delta[1] == -15.0
    pred = mv.prever_B3(pd.DataFrame({"data": pd.to_datetime(["2010-01-01"]),
                                      "vol_pct": [70.0]}), 30, delta)
    assert pred[0] == 55.0      # 70 + (-15)
