# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

import transformadores as tr


def _precip_mensal(anos=40, semente=1):
    rng = np.random.default_rng(semente)
    idx = pd.date_range("1980-01-31", periods=anos * 12, freq="ME")
    # sazonal: mais chuva no verão (DJF), menos no inverno (JJA)
    mes = idx.month
    escala = np.where(np.isin(mes, [12, 1, 2]), 12.0,
             np.where(np.isin(mes, [6, 7, 8]), 2.0, 6.0))
    valores = rng.gamma(1.5, escala)
    return pd.Series(valores, index=idx)


def test_spi_padronizado_no_periodo_de_ajuste():
    mensal = _precip_mensal()
    treino = mensal[mensal.index.year < 2010]

    spi = tr.SPI(escala_meses=3).fit(treino)
    z = spi.transform(treino).dropna()

    assert len(z) > 100
    assert abs(z.mean()) < 0.2, f"média do SPI deveria ~0, veio {z.mean():.3f}"
    assert 0.7 < z.std() < 1.3, f"desvio do SPI deveria ~1, veio {z.std():.3f}"


def test_spi_fit_treino_transform_outro_periodo_sem_erro():
    mensal = _precip_mensal()
    treino = mensal[mensal.index.year < 2010]
    teste = mensal[mensal.index.year >= 2010]
    spi = tr.SPI(escala_meses=6).fit(treino)
    z_teste = spi.transform(teste)
    assert z_teste.notna().sum() > 0
    assert np.isfinite(z_teste.dropna()).all()


def test_spei_lida_com_balanco_negativo():
    rng = np.random.default_rng(3)
    idx = pd.date_range("1980-01-31", periods=40 * 12, freq="ME")
    balanco = pd.Series(rng.normal(-20, 40, len(idx)), index=idx)  # P-ET0, pode ser < 0
    spei = tr.SPEI(escala_meses=12).fit(balanco[balanco.index.year < 2010])
    z = spei.transform(balanco).dropna()
    assert len(z) > 100
    assert np.isfinite(z).all()
    assert abs(z.mean()) < 0.4


def test_escalas_disponiveis():
    mensal = _precip_mensal()
    for k in (3, 6, 12):
        z = tr.SPI(k).fit(mensal).transform(mensal)
        assert z.notna().sum() > 0
