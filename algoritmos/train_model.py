#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_model.py — Treinamento dos modelos de previsão de crise hídrica.

VERSÃO 2 — INCLUI OTIMIZAÇÃO DE THRESHOLD

ATUALIZAÇÃO NESTA VERSÃO:
    Adicionada otimização do threshold de classificação binária. Em
    problemas com forte desbalanceamento de classes (~0.64% de positivos),
    o threshold padrão de 0.5 raramente é atingido, resultando em F1=0
    mesmo quando o modelo apresenta capacidade discriminativa (AUC-ROC).

    A otimização busca, dentro do conjunto de treino via validação
    cruzada temporal, o threshold que maximiza o F1-score. Esse threshold
    é então aplicado ao conjunto de teste.

USO:
    python train_model.py --target y90
    python train_model.py --target y60
    python train_model.py --target y30
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score
)

from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns
import joblib


def setup_logger(level: str = "INFO"):
    """Configura o sistema de logs."""
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S"
    )


def carregar_dados(caminho_parquet: Path, alvo: str = "y90"):
    """
    Carrega o dataset rotulado e prepara X (features) e y (alvo).
    Remove os dias DURANTE a crise (alvo = NaN).
    """
    logging.info(f"Carregando dados de {caminho_parquet}")
    df = pd.read_parquet(caminho_parquet)
    logging.info(f"Dataset bruto: {df.shape[0]:,} linhas × {df.shape[1]} colunas")
    
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    n_antes = len(df)
    df = df.dropna(subset=[alvo]).reset_index(drop=True)
    logging.info(f"Linhas removidas (DURANTE a crise): {n_antes - len(df):,}")
    logging.info(f"Dataset para treino/teste: {len(df):,} linhas")
    
    colunas_excluir = [
        "timestamp", "y30", "y60", "y90",
        "mes", "P30_MEAN_MES", "P30_STD_MES"
    ]
    
    colunas_features = [
        c for c in df.columns
        if c not in colunas_excluir and pd.api.types.is_numeric_dtype(df[c])
    ]
    logging.info(f"Features ({len(colunas_features)}): {colunas_features}")
    
    X = df[colunas_features].copy().fillna(0)
    y = df[alvo].astype(int).copy()
    timestamps = df["timestamp"].copy()
    
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    pct = 100.0 * n_pos / len(y)
    logging.info(
        f"Distribuição {alvo}: positivos={n_pos:,} ({pct:.2f}%) | "
        f"negativos={n_neg:,}"
    )
    
    return X, y, timestamps, colunas_features


def split_temporal(X, y, timestamps, pct_treino: float = 0.8):
    """Split 80/20 respeitando ordem cronológica (sem shuffle)."""
    n_treino = int(len(X) * pct_treino)
    
    X_treino, X_teste = X.iloc[:n_treino], X.iloc[n_treino:]
    y_treino, y_teste = y.iloc[:n_treino], y.iloc[n_treino:]
    ts_treino, ts_teste = timestamps.iloc[:n_treino], timestamps.iloc[n_treino:]
    
    logging.info(
        f"Split: treino={len(X_treino):,} ({ts_treino.min().date()} → "
        f"{ts_treino.max().date()}) | teste={len(X_teste):,} "
        f"({ts_teste.min().date()} → {ts_teste.max().date()})"
    )
    logging.info(
        f"Positivos treino={int(y_treino.sum())} | "
        f"Positivos teste={int(y_teste.sum())}"
    )
    
    return X_treino, X_teste, y_treino, y_teste


def criar_modelos(y_treino):
    """Cria os três modelos com tratamento de desbalanceamento."""
    n_neg = int((y_treino == 0).sum())
    n_pos = int((y_treino == 1).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)
    logging.info(f"scale_pos_weight (XGBoost): {scale_pos_weight:.2f}")
    
    return {
        "DecisionTree": DecisionTreeClassifier(
            max_depth=10, class_weight="balanced", random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=15, class_weight="balanced",
            random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            scale_pos_weight=scale_pos_weight, eval_metric="logloss",
            random_state=42, n_jobs=-1
        )
    }


def otimizar_threshold(modelo, X_treino, y_treino, n_splits=5):
    """
    Encontra o threshold que maximiza F1 via validação cruzada temporal.
    
    Em problemas desbalanceados (0.64% positivos), o threshold padrão
    de 0.5 raramente é cruzado. Esta função busca o threshold ótimo
    em uma grade de 0.01 a 0.99.
    """
    logging.info("Otimizando threshold via TimeSeriesSplit (5 folds)...")
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    thresholds = np.arange(0.01, 1.00, 0.01)
    f1_por_threshold = {t: [] for t in thresholds}
    
    for fold, (idx_tr, idx_val) in enumerate(tscv.split(X_treino), 1):
        modelo_fold = type(modelo)(**modelo.get_params())
        modelo_fold.fit(X_treino.iloc[idx_tr], y_treino.iloc[idx_tr])
        
        y_prob_val = modelo_fold.predict_proba(X_treino.iloc[idx_val])[:, 1]
        y_real_val = y_treino.iloc[idx_val]
        
        for t in thresholds:
            y_pred_t = (y_prob_val >= t).astype(int)
            f1_t = f1_score(y_real_val, y_pred_t, zero_division=0)
            f1_por_threshold[t].append(f1_t)
        
        logging.info(f"  Fold {fold}/{n_splits} concluído")
    
    f1_medio = {t: np.mean(f1_por_threshold[t]) for t in thresholds}
    melhor_threshold = max(f1_medio, key=f1_medio.get)
    melhor_f1_cv = f1_medio[melhor_threshold]
    
    logging.info(
        f"Melhor threshold: {melhor_threshold:.2f} "
        f"(F1 médio CV = {melhor_f1_cv:.4f})"
    )
    
    return melhor_threshold, melhor_f1_cv, f1_medio


def avaliar_modelo(modelo, X_teste, y_teste, nome_modelo: str, threshold: float = 0.5):
    """Avalia o modelo no teste usando o threshold customizado."""
    y_prob = modelo.predict_proba(X_teste)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    
    relatorio = classification_report(
        y_teste, y_pred,
        target_names=["Normal", "Pré-crise"],
        digits=4, zero_division=0
    )
    cm = confusion_matrix(y_teste, y_pred)
    
    try:
        auc = roc_auc_score(y_teste, y_prob)
    except ValueError:
        auc = float("nan")
    
    f1 = f1_score(y_teste, y_pred, zero_division=0)
    
    logging.info(
        f"\n{'='*70}\n[{nome_modelo}] Teste (threshold={threshold:.2f})\n{'='*70}"
    )
    logging.info(f"\n{relatorio}")
    logging.info(f"AUC-ROC: {auc:.4f} | F1: {f1:.4f}")
    logging.info(f"Matriz de confusão:\n{cm}")
    
    return {
        "modelo": nome_modelo,
        "threshold": threshold,
        "f1": f1,
        "auc_roc": auc,
        "matriz_confusao": cm,
        "relatorio": relatorio,
        "y_pred": y_pred,
        "y_prob": y_prob
    }


def plotar_matriz_confusao(cm, nome_modelo, threshold, out_path):
    """Salva matriz de confusão."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Normal", "Pré-crise"],
        yticklabels=["Normal", "Pré-crise"],
        cbar=False
    )
    plt.title(f"Matriz de Confusão — {nome_modelo} (threshold={threshold:.2f})")
    plt.ylabel("Classe Real")
    plt.xlabel("Classe Prevista")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logging.info(f"Matriz salva: {out_path}")


def plotar_importancia_features(modelo, colunas_features, nome_modelo, out_path, top_n=15):
    """Salva gráfico de importância de features."""
    if not hasattr(modelo, "feature_importances_"):
        return
    
    importancias = pd.Series(modelo.feature_importances_, index=colunas_features)
    importancias = importancias.sort_values(ascending=True).tail(top_n)
    
    plt.figure(figsize=(10, max(5, top_n * 0.35)))
    importancias.plot(kind="barh", color="#1D9E75")
    plt.title(f"Top {top_n} Features — {nome_modelo}")
    plt.xlabel("Importância relativa")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logging.info(f"Importância salva: {out_path}")


def plotar_curva_threshold(f1_por_threshold, nome_modelo, melhor_threshold, out_path):
    """Plota F1 vs threshold (visualiza o efeito da otimização)."""
    thresholds = sorted(f1_por_threshold.keys())
    f1_valores = [f1_por_threshold[t] for t in thresholds]
    
    plt.figure(figsize=(10, 5))
    plt.plot(thresholds, f1_valores, color="#1D9E75", linewidth=2)
    plt.axvline(x=melhor_threshold, color="#D85A30", linestyle="--",
                label=f"Melhor threshold = {melhor_threshold:.2f}")
    plt.title(f"F1-score vs Threshold — {nome_modelo}")
    plt.xlabel("Threshold de classificação")
    plt.ylabel("F1-score (média CV)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logging.info(f"Curva threshold salva: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="algoritmos/data/features/inmet_sp_daily_labels.parquet")
    ap.add_argument("--target", default="y90", choices=["y30", "y60", "y90"])
    ap.add_argument("--out-dir", default="resultados")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    
    setup_logger(args.log_level)
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp_run = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.info(f"=== INÍCIO | alvo={args.target} ===")
    
    X, y, timestamps, colunas_features = carregar_dados(Path(args.data), alvo=args.target)
    X_treino, X_teste, y_treino, y_teste = split_temporal(X, y, timestamps)
    modelos = criar_modelos(y_treino)
    
    resultados = []
    
    for nome, modelo in modelos.items():
        logging.info(f"\n{'#'*70}\n# {nome}\n{'#'*70}")
        
        # Treina no conjunto completo
        logging.info(f"[{nome}] Treinando...")
        modelo.fit(X_treino, y_treino)
        
        # Otimiza threshold
        modelo_para_cv = type(modelo)(**modelo.get_params())
        melhor_thr, f1_cv, f1_por_thr = otimizar_threshold(
            modelo_para_cv, X_treino, y_treino
        )
        
        # Avalia no teste
        resultado = avaliar_modelo(modelo, X_teste, y_teste, nome, threshold=melhor_thr)
        resultado["f1_cv_otimo"] = f1_cv
        resultado["modelo_obj"] = modelo
        resultados.append(resultado)
        
        # Gráficos
        plotar_matriz_confusao(
            resultado["matriz_confusao"], nome, melhor_thr,
            out_dir / f"matriz_confusao_{nome}_{args.target}.png"
        )
        plotar_importancia_features(
            modelo, colunas_features, nome,
            out_dir / f"importancia_features_{nome}_{args.target}.png"
        )
        plotar_curva_threshold(
            f1_por_thr, nome, melhor_thr,
            out_dir / f"curva_threshold_{nome}_{args.target}.png"
        )
    
    # Melhor modelo
    melhor = max(resultados, key=lambda r: r["f1"])
    logging.info(
        f"\n{'='*70}\nMELHOR MODELO: {melhor['modelo']} "
        f"(threshold={melhor['threshold']:.2f}, F1={melhor['f1']:.4f}, "
        f"AUC-ROC={melhor['auc_roc']:.4f})\n{'='*70}"
    )
    
    caminho_modelo = out_dir / f"modelo_crise_hidrica_{args.target}.pkl"
    joblib.dump(
        {"modelo": melhor["modelo_obj"], "threshold": melhor["threshold"]},
        caminho_modelo
    )
    logging.info(f"Modelo + threshold salvos: {caminho_modelo}")
    
    # Tabela comparativa
    tabela = pd.DataFrame([
        {
            "Modelo": r["modelo"],
            "Threshold": round(r["threshold"], 2),
            "F1 (teste)": round(r["f1"], 4),
            "AUC-ROC (teste)": round(r["auc_roc"], 4),
            "F1 (CV)": round(r["f1_cv_otimo"], 4),
        }
        for r in resultados
    ])
    caminho_tabela = out_dir / f"comparativo_modelos_{args.target}_{timestamp_run}.csv"
    tabela.to_csv(caminho_tabela, index=False, encoding="utf-8")
    logging.info(f"\n{tabela.to_string(index=False)}")
    logging.info(f"Tabela salva: {caminho_tabela}")
    
    logging.info("=== FIM | OK ===")


if __name__ == "__main__":
    main()