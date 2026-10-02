# -*- coding: utf-8 -*-
"""
lstm_v2.py — LSTM (PyTorch CPU) no MESMO arcabouço do Exp 3, para comparação direta.

- Mesmos 4 folds, mesma purga (pela maior janela, h=90), transformadores/escalonador
  ajustados SÓ no treino de cada fold.
- Alvo: o RESÍDUO sobre o B3 (r_30, r_60, r_90) — igual ao XGBRes.
- Entrada: janelas dos últimos `lookback` (=180) dias das features de CLIMA_ESTADO,
  escalonadas (StandardScaler fit no treino do fold). A janela termina em t (nada > t).
- Arquitetura: 1 camada LSTM (hidden 32/64) + dropout + Linear(hidden, 3) (uma saída
  por horizonte). Early stopping no trecho final cronológico (20%) do treino do fold.
- 3 seeds, média das previsões. Orçamento de tempo global (budget_s).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

import modelagem_v2 as mv

LOOKBACK = 180
HORIZONTES = (30, 60, 90)
FEATS = mv.CLIMA_ERA5 + mv.ESTADO       # features diárias de CLIMA_ESTADO (sem SPI, que é por-fold)


class LSTMResidual(nn.Module):
    def __init__(self, n_feat, hidden=32, n_saidas=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, n_saidas)

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.fc(self.drop(o[:, -1, :]))      # usa o estado em t (fim da janela)


def _janelas(M, posicoes, lookback=LOOKBACK):
    """Empilha janelas [n, lookback, F] terminando em cada posição (inclui t)."""
    idx = posicoes[:, None] - np.arange(lookback - 1, -1, -1)[None, :]
    return M[idx]


def _posicoes_validas(M, mask_alvo, lookback=LOOKBACK):
    """Posições (linhas) com alvo marcado, janela completa e sem NaN na janela."""
    pos = np.where(mask_alvo)[0]
    pos = pos[pos >= lookback - 1]
    if len(pos) == 0:
        return pos
    jj = _janelas(M, pos, lookback)
    ok = ~np.isnan(jj).any(axis=(1, 2))
    return pos[ok]


def _treinar_uma(Xtr, Ytr, Xiv, Yiv, n_feat, hidden, dropout, seed, max_epocas=60,
                 paciencia=8, lr=1e-3, batch=64):
    torch.manual_seed(seed)
    modelo = LSTMResidual(n_feat, hidden=hidden, dropout=dropout)
    opt = torch.optim.Adam(modelo.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.MSELoss()
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32); Ytr_t = torch.tensor(Ytr, dtype=torch.float32)
    Xiv_t = torch.tensor(Xiv, dtype=torch.float32); Yiv_t = torch.tensor(Yiv, dtype=torch.float32)
    n = len(Xtr_t); melhor, melhor_estado, espera = np.inf, None, 0
    for _ in range(max_epocas):
        modelo.train(); perm = torch.randperm(n)
        for i in range(0, n, batch):
            j = perm[i:i + batch]
            opt.zero_grad()
            loss = lossf(modelo(Xtr_t[j]), Ytr_t[j])
            loss.backward(); opt.step()
        modelo.eval()
        with torch.no_grad():
            vl = lossf(modelo(Xiv_t), Yiv_t).item()
        if vl < melhor - 1e-4:
            melhor, melhor_estado, espera = vl, {k: v.clone() for k, v in modelo.state_dict().items()}, 0
        else:
            espera += 1
            if espera >= paciencia:
                break
    if melhor_estado is not None:
        modelo.load_state_dict(melhor_estado)
    modelo.eval()
    return modelo


def validar_lstm(df, cfg, hidden=32, dropout=0.2, seeds=(0, 1, 2), stride_treino=5,
                 budget_s=1500, log=print):
    """Valida o LSTM residual em todos os folds/horizontes. Retorna (linhas, oof_por_h, info)."""
    t0 = time.time()
    folds = mv.folds_config(cfg)
    datas = pd.to_datetime(df["data"])
    M_full = df[FEATS].to_numpy(dtype=np.float64)
    completo = True
    linhas = []
    oof_acc = {h: {"data": [], "real": [], "pred": []} for h in HORIZONTES}

    for fo in folds:
        if time.time() - t0 > budget_s:
            log(f"[LSTM] orçamento ({budget_s}s) esgotado antes do fold {fo['nome']}.")
            completo = False
            break
        # purga pela MAIOR janela (h=90); val = bloco
        trm90, vam = mv.mascaras_fold(df["data"], fo["ini"], fo["fim"], 90)
        # escalonador: fit nas features do REGIÃO de treino (datas < início do fold)
        treino_regiao = (datas < pd.Timestamp(fo["ini"])).values
        sc = StandardScaler().fit(M_full[treino_regiao & ~np.isnan(M_full).any(axis=1)])
        M = sc.transform(np.nan_to_num(M_full, nan=np.nan))  # mantém NaN p/ filtro de janela
        M = np.where(np.isnan(M_full), np.nan, M)

        # alvos r_h por fold (B3 ajustado no treino)
        alvo_cols = {}
        for h in HORIZONTES:
            delta = mv.ajustar_B3(df[trm90 & df[f"vol_pct_t{h}"].notna().values], h)
            b3_all = mv.prever_B3(df, h, delta)
            alvo_cols[h] = df[f"vol_pct_t{h}"].to_numpy() - b3_all   # r_h em todas as linhas

        alvo_ok = np.all([~np.isnan(alvo_cols[h]) for h in HORIZONTES], axis=0)
        pos_tr = _posicoes_validas(M, trm90 & alvo_ok)
        pos_va = _posicoes_validas(M, vam & alvo_ok)
        if len(pos_tr) < 100 or len(pos_va) < 20:
            log(f"[LSTM] fold {fo['nome']}: amostras insuficientes, pulando.")
            continue
        pos_tr = pos_tr[::stride_treino]

        Y = np.column_stack([alvo_cols[h] for h in HORIZONTES])
        # padroniza alvos (fit no treino) p/ estabilidade
        ty_mean = Y[pos_tr].mean(axis=0); ty_std = Y[pos_tr].std(axis=0) + 1e-6
        corte = int(len(pos_tr) * 0.8)                       # val interna = 20% final cronológico
        p_fit, p_iv = pos_tr[:corte], pos_tr[corte:]
        Xtr = _janelas(M, p_fit); Xiv = _janelas(M, p_iv); Xva = _janelas(M, pos_va)
        Ytr = (Y[p_fit] - ty_mean) / ty_std; Yiv = (Y[p_iv] - ty_mean) / ty_std

        preds_seeds = []
        for s in seeds:
            if time.time() - t0 > budget_s:
                completo = False; break
            mod = _treinar_uma(Xtr, Ytr, Xiv, Yiv, len(FEATS), hidden, dropout, s)
            with torch.no_grad():
                p = mod(torch.tensor(Xva, dtype=torch.float32)).numpy()
            preds_seeds.append(p * ty_std + ty_mean)         # volta à escala do resíduo
        if not preds_seeds:
            completo = False; break
        r_pred = np.mean(preds_seeds, axis=0)                # [n_va, 3] resíduos previstos

        for hi, h in enumerate(HORIZONTES):
            delta = mv.ajustar_B3(df[trm90 & df[f"vol_pct_t{h}"].notna().values], h)
            b3_va = mv.prever_B3(df.iloc[pos_va], h, delta)
            real = df.iloc[pos_va][f"vol_pct_t{h}"].to_numpy()
            b1 = df.iloc[pos_va]["vol_pct"].to_numpy()
            pred_vol = b3_va + r_pred[:, hi]
            mae = np.mean(np.abs(real - pred_vol))
            mae_b1 = np.mean(np.abs(real - b1)); mae_b3 = np.mean(np.abs(real - b3_va))
            linhas.append({"fold": fo["nome"], "modelo": "LSTM", "horizonte": h,
                           "MAE": round(float(mae), 3),
                           "skill_B1": round(1 - mae / mae_b1, 4) if mae_b1 > 0 else np.nan,
                           "skill_B3": round(1 - mae / mae_b3, 4) if mae_b3 > 0 else np.nan})
            oof_acc[h]["data"].append(df.iloc[pos_va]["data"].to_numpy())
            oof_acc[h]["real"].append(real); oof_acc[h]["pred"].append(pred_vol)
        log(f"[LSTM] fold {fo['nome']} ok ({time.time()-t0:.0f}s)")

    oof_por_h = {}
    for h in HORIZONTES:
        if oof_acc[h]["data"]:
            oof_por_h[h] = pd.DataFrame({
                "data": np.concatenate(oof_acc[h]["data"]),
                "real": np.concatenate(oof_acc[h]["real"]),
                "LSTM": np.concatenate(oof_acc[h]["pred"])}).sort_values("data").reset_index(drop=True)
    info = {"completo": completo, "tempo_s": round(time.time() - t0, 1),
            "hidden": hidden, "seeds": list(seeds), "stride_treino": stride_treino}
    return linhas, oof_por_h, info
