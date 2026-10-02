#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, time
from pathlib import Path
import pandas as pd, numpy as np
import logging
"""
03_build_labels.py — Gera y30/y60/y90 a partir de eventos (pré-crise).
Etapa 3 do pipeline.
Uso:
  python pipeline/03_build_labels.py --features data/features/inmet_sp_daily_features.parquet \
    --events data/events/eventos.csv --out data/features/inmet_sp_daily_labels.parquet \
    --system-col sistema_alvo --default-system Cantareira --exclude-during --log-level INFO
"""

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
def setup_logger(level: str):
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=lvl, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
def load_events(path):
    p=Path(path)
    logging.info(f"Lendo eventos: {p}")
    if p.suffix.lower()=='.xlsx':
        ev=pd.read_excel(p, engine='openpyxl')
    else:
        ev=pd.read_csv(p)
    ev.columns=[str(c).strip().lower() for c in ev.columns]
    if 'sistema' not in ev.columns or 'start_date' not in ev.columns:
        raise KeyError("Eventos precisam de colunas: sistema, start_date[, end_date, criterio]")
    ev['start_date']=pd.to_datetime(ev['start_date'], errors='coerce').dt.tz_localize('UTC')
    if 'end_date' in ev.columns:
        ev['end_date']=pd.to_datetime(ev['end_date'], errors='coerce').dt.tz_localize('UTC')
    else:
        ev['end_date']=pd.NaT
    logging.info(f"Eventos carregados: {len(ev)}")
    logging.info(f"Amostra:\n{ev.head(5)}")
    return ev
def label_pre_evento(df, eventos, horizons=(30,60,90), w_pre=7, w_pos=7,
                     system_col=None, default_system=None, exclude_during=False):
    t0 = time.time()
    out=df.copy()
    out['timestamp']=pd.to_datetime(out['timestamp'], utc=True)
    if system_col is None or system_col not in out.columns:
        system_col='_system_tmp_'; out[system_col]=default_system or 'GLOBAL'
        logging.warning(f"system_col não fornecida/encontrada. Usando '{system_col}'='{default_system or 'GLOBAL'}' para todas as linhas.")
    for h in horizons:
        out[f'y{h}']=0
    logging.info(f"Rotulando com horizontes {list(horizons)}, tolerância w_pre={w_pre}, w_pos={w_pos}")
    for i, ev in eventos.iterrows():
        sysname=ev['sistema']; t0_ev=ev['start_date']; t1_ev=ev.get('end_date', pd.NaT)
        if pd.isna(t0_ev): 
            logging.warning(f"Evento {i} sem start_date válido. Skip.")
            continue
        logging.info(f"Evento {i}: sistema={sysname} start={t0_ev.date()} end={str(t1_ev.date()) if pd.notna(t1_ev) else '—'}")
        mask_sys=(out[system_col]==sysname)
        for h in horizons:
            win_ini=t0_ev - pd.Timedelta(days=h + w_pre)
            win_fim=t0_ev - pd.Timedelta(days=h - w_pos)
            mask_time=(out['timestamp']>=win_ini)&(out['timestamp']<=win_fim)
            n = int((mask_sys & mask_time).sum())
            out.loc[mask_sys & mask_time, f'y{h}']=1
            logging.info(f"  y{h}: janela [{win_ini.date()} .. {win_fim.date()}] -> marcados={n}")
        if exclude_during and pd.notna(t1_ev):
            mask_dur=(out['timestamp']>=t0_ev)&(out['timestamp']<=t1_ev)
            out.loc[mask_sys & mask_dur, [f'y{h}' for h in horizons]]=np.nan
            logging.info(f"  Período do evento excluído do treino: {(mask_sys & mask_dur).sum()} linhas -> NaN")
    logging.info(f"Rotulagem concluída em {time.time()-t0:.1f}s")
    return out
def label_stats(df, horizons=(30,60,90)):
    rows=[]
    for h in horizons:
        col=f'y{h}'
        if col in df.columns:
            s=df[col].dropna(); pos=int((s==1).sum()); neg=int((s==0).sum())
            total=pos+neg; pct=100.0*pos/max(total,1)
            rows.append({'label':col,'positives':pos,'negatives':neg,'pct_positive':round(pct,4)})
    return pd.DataFrame(rows)
def human_size(path: Path):
    try:
        b = path.stat().st_size
        for unit in ['B','KB','MB','GB']:
            if b < 1024:
                return f"{b:.1f}{unit}"
            b /= 1024
        return f"{b:.1f}TB"
    except Exception:
        return "N/A"
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--features', default=str(DATA / 'features' / 'inmet_sp_daily_features.parquet'))
    ap.add_argument('--events', default=str(DATA / 'events' / 'eventos.csv'))
    ap.add_argument('--out', default=str(DATA / 'features' / 'inmet_sp_daily_labels.parquet'))
    ap.add_argument('--system-col', default=None)
    ap.add_argument('--default-system', default=None)
    ap.add_argument('--exclude-during', action='store_true')
    ap.add_argument('--w-pre', type=int, default=7)
    ap.add_argument('--w-pos', type=int, default=7)
    ap.add_argument('--log-level', default='INFO')
    args=ap.parse_args()
    setup_logger(args.log_level)
    logging.info("=== INÍCIO LABELS ===")
    logging.info(f"Lendo features: {args.features}")
    df=pd.read_parquet(args.features)
    logging.info(f"Features shape={df.shape}")
    eventos=load_events(args.events)
    labeled=label_pre_evento(df, eventos, horizons=(30,60,90),
                             w_pre=args.w_pre, w_pos=args.w_pos,
                             system_col=args.system_col, default_system=args.default_system,
                             exclude_during=args.exclude_during)
    out=Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_parquet(out, index=False)
    stats=label_stats(labeled); stats_path=out.with_suffix('').as_posix()+'_stats.csv'
    stats.to_csv(stats_path, index=False, encoding='utf-8')
    logging.info(f"Arquivos salvos: {out} ({human_size(out)}) e {stats_path}")
    logging.info("=== FIM LABELS | OK ===")
if __name__=='__main__':
    main()
