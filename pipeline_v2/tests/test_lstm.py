# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import lstm_v2 as lv


# --------------------------------------- a janela termina em t (nada depois de t entra)
def test_janela_termina_em_t_e_e_causal():
    T, F = 500, 3
    # feature = posição no tempo, p/ detectar qualquer "vazamento" do futuro
    M = np.tile(np.arange(T).reshape(-1, 1), (1, F)).astype(float)
    pos = np.array([200, 300, 499])
    jj = lv._janelas(M, pos, lookback=lv.LOOKBACK)
    assert jj.shape == (3, lv.LOOKBACK, F)
    for k, p in enumerate(pos):
        assert jj[k, -1, 0] == p, "o último passo da janela não é t"
        assert jj[k].max() == p, "entrou informação de depois de t"
        assert jj[k, 0, 0] == p - lv.LOOKBACK + 1

    # alterar o FUTURO (t+5) não pode mudar a janela que termina em t
    p = 300
    j_antes = lv._janelas(M, np.array([p]))[0].copy()
    M2 = M.copy(); M2[p + 5:] += 999.0
    j_depois = lv._janelas(M2, np.array([p]))[0]
    assert np.array_equal(j_antes, j_depois)


# --------------------------------------- escalonador ajustado só no treino
def test_escalonador_ajustado_so_no_treino():
    T, F = 1000, 4
    rng = np.random.default_rng(0)
    M = rng.normal(0, 1, (T, F))
    datas = pd.date_range("2000-01-01", periods=T, freq="D")
    treino_regiao = np.asarray(datas < pd.Timestamp("2002-01-01"))   # como em validar_lstm

    sc1 = StandardScaler().fit(M[treino_regiao])
    M2 = M.copy()
    M2[~treino_regiao] *= 50.0                                     # altera só a validação/futuro
    sc2 = StandardScaler().fit(M2[treino_regiao])

    assert np.allclose(sc1.mean_, sc2.mean_), "média do escalonador mudou com a validação"
    assert np.allclose(sc1.scale_, sc2.scale_), "escala do escalonador mudou com a validação"
