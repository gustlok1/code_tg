#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_build_features.py — Agrega hora→dia, cria features (variáveis de entrada) e SPI30 aproximado.
Etapa 2 do pipeline.
Uso:
  python pipeline/02_build_features.py --ints data/interim/inmet_sp_hourly_clean.parquet --out-dir data/features --log-level INFO
  # Se existir coluna de estação no parquet e você quiser agrupar por estação:
  # --station-col NOME_DA_COLUNA
"""
import argparse, re
from pathlib import Path
import pandas as pd, numpy as np
import logging

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# ---------- logging ----------
def setup_logger(level: str):
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=lvl, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")

# ---------- util ----------
def guess_station_col(cols):
    """Heurística para achar coluna de estação (se existir)."""
    pats=[r'^ESTA[CÇ][AÃ]O', r'COD.*ESTA', r'ESTATION|STATION|ID_?ESTA', r'^WMO$', r'^ID$']
    for pat in pats:
        rx=re.compile(pat, re.IGNORECASE)
        for c in cols:
            if rx.search(str(c)): return c
    return None

def _apply_by(df, keys, col, func):
    """
    Aplica 'func' na série df[col] com/sem agrupamento.
    Evita erro 'No group keys passed!' quando keys=[].
    """
    if keys:
        return df.groupby(keys, group_keys=False)[col].apply(func)
    else:
        return func(df[col])

# ---------- hora -> dia ----------
def hourly_to_daily(df, station_col=None):
    """
    Converte hora→dia. Se houver coluna de estação, agrega por [estacao × dia],
    senão agrega global por dia.
    - groupby (agrupar por chave/categoria)
    - resample (reamostrar por tempo, ex.: 'D' = diário)
    - Grouper (agrupador temporal do pandas)
    """
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    cols = df.columns

    # nomes esperados (INMET)
    p     = 'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)'
    t     = 'TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)'
    tmaxh = 'TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)'
    tminh = 'TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)'
    urh   = 'UMIDADE RELATIVA DO AR, HORARIA (%)'
    rad   = 'RADIACAO GLOBAL (KJ/m²)'
    press = 'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)'
    vmed  = 'VENTO, VELOCIDADE HORARIA (m/s)'
    vraj  = 'VENTO, RAJADA MAXIMA (m/s)'

    aggs = {
        p: 'sum',
        t: 'mean',
        tmaxh: 'max',
        tminh: 'min',
        urh: 'mean',
        rad: 'sum',
        press: 'mean',
        vmed: 'mean',
        vraj: 'max',
    }
    aggs = {k: v for k, v in aggs.items() if k in cols}
    if not aggs:
        raise ValueError("Nenhuma coluna esperada para agregação encontrada no DF.\n"
                         f"Colunas disponíveis: {list(cols)}")

    logging.info(f"Colunas agregadas (hora→dia): {list(aggs.keys())}")

    # índice temporal para resample
    df = df.set_index('timestamp')

    by_station = bool(station_col) and (station_col in cols)
    if by_station:
        logging.info(f"Agrupando por estação '{station_col}' + dia (Grouper diário).")
        daily = (
            df.groupby([station_col, pd.Grouper(freq='D')])
              .agg(aggs)
              .reset_index()
        )
        # garantir nome 'timestamp' para a coluna de data
        if 'timestamp' not in daily.columns:
            for c in daily.columns:
                if pd.api.types.is_datetime64_any_dtype(daily[c]):
                    daily = daily.rename(columns={c: 'timestamp'})
                    break
    else:
        logging.info("Sem coluna de estação — agregando global por dia (resample).")
        daily = (
            df.resample('D')
              .agg(aggs)
              .reset_index()
        )

    # renomear para nomes amigáveis
    rename = {
        p: 'PRECIP_DIARIA',
        t: 'TMEAN',
        tmaxh: 'TMAX',
        tminh: 'TMIN',
        urh: 'URMEAN',
        rad: 'RAD_SUM',
        press: 'PRESSAO_MED',
        vmed: 'VENTO_MED',
        vraj: 'RAJADA_MAX',
    }
    daily = daily.rename(columns={k: v for k, v in rename.items() if k in daily.columns})
    logging.info(f"Diário gerado: shape={daily.shape}")
    return daily

# ---------- features ----------
def add_features_and_spi(daily, station_col=None):
    """
    Cria features derivadas:
      - P7/P15/P30/P90 (acúmulos — últimos N dias, janela móvel)
      - DRY_STREAK_CUR, DRY30_MAX, DRY90_MAX (sequências secas)
      - Médias móveis: TMEAN_MA7/MA30, URMEAN_MA7/MA30, PRESSAO_MED_MA7
      - SPI30_APRX (z-score mensal de P30; aproxima o SPI de 30 dias)
    """
    daily = daily.copy()
    if 'timestamp' not in daily.columns:
        raise ValueError("A coluna 'timestamp' é obrigatória.")
    daily['timestamp'] = pd.to_datetime(daily['timestamp'], utc=True)

    grp = [station_col] if (station_col and station_col in daily.columns) else []
    daily = daily.sort_values(grp+['timestamp']) if grp else daily.sort_values('timestamp')

    # P7/P15/P30/P90
    if 'PRECIP_DIARIA' in daily.columns:
        logging.info("Calculando P7/P15/P30/P90 (acúmulos).")

        def roll_sum(s, w):  # janela móvel (últimos N dias)
            return s.rolling(w, min_periods=int(w*0.7)).sum()

        for w in [7, 15, 30, 90]:
            daily[f'P{w}'] = _apply_by(daily, grp, 'PRECIP_DIARIA', lambda s: roll_sum(s, w))
    else:
        logging.warning("PRECIP_DIARIA não encontrada — P*/DRY* não serão criados.")

    # dry spells
    if 'PRECIP_DIARIA' in daily.columns:
        logging.info("Calculando sequências secas (dry spells).")
        def dry_streak(s):
            is_dry = (s < 1.0).astype(float)
            c = (is_dry == 0).cumsum()
            return is_dry.groupby(c).cumcount() + is_dry

        daily['DRY_STREAK_CUR'] = _apply_by(daily, grp, 'PRECIP_DIARIA', dry_streak)

        def roll_max(s, w, mp):
            return s.rolling(w, min_periods=mp).max()

        daily['DRY30_MAX'] = _apply_by(daily, grp, 'DRY_STREAK_CUR', lambda s: roll_max(s, 30, 21))
        daily['DRY90_MAX'] = _apply_by(daily, grp, 'DRY_STREAK_CUR', lambda s: roll_max(s, 90, 63))

    # médias móveis
    logging.info("Calculando médias móveis (TMEAN/URMEAN/PRESSAO_MED).")
    def roll_mean(s, w):  # média móvel
        return s.rolling(w, min_periods=int(w*0.7)).mean()

    for base, w in [('TMEAN',7), ('TMEAN',30), ('URMEAN',7), ('URMEAN',30), ('PRESSAO_MED',7)]:
        if base in daily.columns:
            daily[f'{base}_MA{w}'] = _apply_by(daily, grp, base, lambda s: roll_mean(s, w))

    # SPI30 aproximado (z-score mensal do P30)
    logging.info("Calculando SPI30_APRX (z-score mensal de P30).")
    daily['mes'] = daily['timestamp'].dt.month
    if 'P30' in daily.columns:
        keys = (grp + ['mes']) if grp else ['mes']
        clim = (daily.dropna(subset=['P30'])
                    .groupby(keys)['P30']
                    .agg(['mean','std'])
                    .rename(columns={'mean':'P30_MEAN_MES','std':'P30_STD_MES'})
                    .reset_index())
        daily = daily.merge(clim, on=keys, how='left')
        daily['SPI30_APRX'] = (daily['P30'] - daily['P30_MEAN_MES']) / daily['P30_STD_MES']
    else:
        daily['SPI30_APRX'] = np.nan
        logging.warning("P30 ausente — SPI30_APRX ficou NaN.")

    logging.info(f"Features criadas: shape={daily.shape}")
    return daily

# ---------- io ----------
def human_size(path: Path):
    try:
        b = path.stat().st_size
        for unit in ['B','KB','MB','GB']:
            if b < 1024: return f"{b:.1f}{unit}"
            b /= 1024
        return f"{b:.1f}TB"
    except Exception:
        return "N/A"

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ints', default=str(DATA / 'interim' / 'inmet_sp_hourly_clean.parquet'))
    ap.add_argument('--out-dir', default=str(DATA / 'features'))
    ap.add_argument('--station-col', default=None)
    ap.add_argument('--log-level', default='INFO')
    args = ap.parse_args()
    setup_logger(args.log_level)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    logging.info("=== INÍCIO FEATURES ===")
    logging.info(f"Lendo {args.ints} ...")
    df = pd.read_parquet(args.ints)
    logging.info(f"Carregado: shape={df.shape}")

    station_col = args.station_col or guess_station_col(df.columns)
    if station_col and station_col in df.columns:
        logging.info(f"Coluna de estação: {station_col} (n únicos = {df[station_col].nunique()})")
    else:
        station_col = None
        logging.info("Sem coluna de estação — processamento global.")

    daily = hourly_to_daily(df, station_col)
    daily_out = out_dir/'inmet_sp_daily.parquet'
    daily.to_parquet(daily_out, index=False)
    logging.info(f"Salvo diário: {daily_out} ({human_size(daily_out)})")

    daily_feat = add_features_and_spi(daily, station_col)
    feat_out = out_dir/'inmet_sp_daily_features.parquet'
    daily_feat.to_parquet(feat_out, index=False)
    logging.info(f"Salvo features: {feat_out} ({human_size(feat_out)})")
    logging.info("=== FIM FEATURES | OK ===")

if __name__ == '__main__':
    main()
