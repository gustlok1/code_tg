# -*- coding: utf-8 -*-
"""Prova que a faixa de incerteza da pagina "Situacao atual" vem das previsoes FORA DA
AMOSTRA da validacao (OOF do XGBRes, 1984-2022), e NAO dos residuos de treino."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "app_v2"))
sys.path.insert(0, str(ROOT / "pipeline_v2"))

import analise_exp3 as ax       # noqa: E402
import modelagem_v2 as mv       # noqa: E402
import utils_v2 as u            # noqa: E402

DATASET = ROOT / "data" / "processed_v2" / "dataset_diario.parquet"
H = 30


@pytest.mark.skipif(not DATASET.exists(), reason="dataset v2 ainda não gerado")
def test_faixa_vem_da_validacao_oof_nao_do_treino():
    df = u.carregar_dataset()
    c = u.cfg()

    # 1) a faixa do app e o percentil 80 do erro ABSOLUTO OOF da validacao (deterministico)
    _, oof = ax.validar_residual(df, c, H, src="era5")
    esperado_oof = float(np.percentile(np.abs(oof["real"].values - oof["XGBRes"].values), 80))
    banda_app = u.faixa_incerteza_validacao(df, H)
    assert banda_app == pytest.approx(esperado_oof, abs=1e-9)

    # 2) a mesma faixa calculada com os residuos de TREINO (dentro da amostra) e bem menor;
    #    logo a faixa do app NAO vem do treino.
    alvo = f"vol_pct_t{H}"
    mask_fit = df[alvo].notna().values & df["vol_pct"].notna().values
    X, cols = u._features(df, mask_fit)
    ok = X[cols].notna().all(axis=1).values
    tr_ok = ok & df[alvo].notna().values
    delta = mv.ajustar_B3(df[tr_ok], H)
    b3_tr = mv.prever_B3(df[tr_ok], H, delta)
    r_tr = df.loc[tr_ok, alvo].values - b3_tr
    from xgboost import XGBRegressor
    config = u.congelamento()["modelos"]["XGBRes"].get(str(H), {}).get("config", {})
    mod = XGBRegressor(**(config or {}), random_state=42, n_jobs=-1, objective="reg:squarederror")
    mod.fit(X[tr_ok], r_tr)
    resid_treino = df.loc[tr_ok, alvo].values - (b3_tr + mod.predict(X[tr_ok]))
    banda_treino = float(np.percentile(np.abs(resid_treino), 80))
    assert banda_app > banda_treino + 0.3              # OOF sensivelmente maior que dentro da amostra
    assert banda_app != pytest.approx(banda_treino, abs=1e-6)

    # 3) a previsao usa a faixa INJETADA (da validacao), nao uma recalculada no treino
    _, banda_usada = u.treinar_e_prever(df, H, config, banda=esperado_oof)
    assert banda_usada == pytest.approx(esperado_oof, abs=1e-12)
