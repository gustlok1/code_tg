#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, re, time
from pathlib import Path
import pandas as pd, numpy as np
import logging
"""
run_ints_xlsx.py — Limpa e unifica arquivos .xlsx do INMET (horários) e gera Parquet + QC.
Uso:
  python run_ints_xlsx.py --raw-dir data/raw --out-dir data/ints --log-level INFO
Dependências:
  pip install pandas numpy pyarrow openpyxl
"""
def setup_logger(level: str):
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=lvl, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
def find_col(columns, patterns):
    for pat in patterns:
        rx = re.compile(pat, re.IGNORECASE)
        for c in columns:
            if rx.search(str(c).strip()):
                return c
    return None
def coerce_numeric(series, colname):
    s = series.astype(str).str.strip()
    s = s.replace({'-':np.nan,'':np.nan,'nan':np.nan,'None':np.nan,'NaN':np.nan,
                   '-9999':np.nan,'-9999.0':np.nan,'-9999,0':np.nan})
    s = s.str.replace(',', '.', regex=False)
    s = s.str.replace(r'\s+','', regex=True)
    out = pd.to_numeric(s, errors='coerce')
    logging.debug(f"[NUM] {colname}: NaNs={out.isna().sum():,}")
    return out
def read_all_xlsx(raw_dir: Path) -> pd.DataFrame:
    files = sorted(raw_dir.rglob('*.xlsx'))
    if not files:
        raise FileNotFoundError(f"Nenhum .xlsx encontrado em {raw_dir}")
    logging.info(f"Encontrados {len(files)} arquivos .xlsx em {raw_dir}")
    for i, f in enumerate(files[:10], 1):
        logging.info(f"  {i:02d}) {f}")
    if len(files) > 10:
        logging.info(f"  ... (+{len(files)-10} outros)")
    dfs = []
    total_rows = 0
    t0 = time.time()
    for idx, f in enumerate(files, 1):
        try:
            book = pd.read_excel(f, sheet_name=None, dtype=object, engine='openpyxl')
            if not book:
                logging.warning(f"[VAZIO] {f} sem planilhas")
                continue
            for sheet_name, df in book.items():
                if df is None or df.empty:
                    logging.debug(f"[VAZIO] {f}::{sheet_name} sem dados")
                    continue
                df.columns = [str(c).strip() for c in df.columns]
                _dc = find_col(df.columns, [r'^DATA\b', r'^Data$'])
                _hc = find_col(df.columns, [r'^HORA', r'^Hora UTC$'])
                _rn = {}
                if _dc and _dc != 'DATA (YYYY-MM-DD)': _rn[_dc] = 'DATA (YYYY-MM-DD)'
                if _hc and _hc != 'HORA (UTC)': _rn[_hc] = 'HORA (UTC)'
                if _rn: df = df.rename(columns=_rn)
                df['_file'] = str(f)
                df['_sheet'] = str(sheet_name)
                total_rows += len(df)
                dfs.append(df)
            if idx % 5 == 0:
                logging.info(f"Lido {idx}/{len(files)} arquivos... linhas acumuladas ~ {total_rows:,}")
        except Exception as e:
            logging.exception(f"Falha ao ler {f}: {e}")
            raise
    big = pd.concat(dfs, ignore_index=True)
    logging.info(f"Concatenação concluída: shape={big.shape}, tempo={time.time()-t0:.1f}s")
    return big
def clean_and_unify_xlsx(raw_dir: Path) -> pd.DataFrame:
    df = read_all_xlsx(raw_dir)
    logging.info(f"Colunas detectadas: {list(df.columns)[:12]}{' ...' if len(df.columns)>12 else ''}")
    data_col = find_col(df.columns, [r'^DATA\b', r'^Data\b'])
    hora_col = find_col(df.columns, [r'^HORA', r'HORA.*UTC', r'^Hora'])
    logging.info(f"Coluna DATA: {data_col} | Coluna HORA: {hora_col}")
    if data_col is None or hora_col is None:
        raise KeyError("Esperadas colunas tipo 'DATA' e 'HORA (UTC)'.")
    df[data_col] = df[data_col].astype(str).str.strip().str.replace('/', '-', regex=False)
    h = df[hora_col].astype(str).str.strip()
    h = h.str.replace(r'\s*UTC$', '', regex=True)
    h = h.str.replace(r'^(\d{2})(\d{2})$', r'\1:\2', regex=True)
    h = h.str.replace(r'^\s*$', '00:00', regex=True)
    h = h.str.replace(r'^\d{1}$', lambda m: m.group(0).zfill(2)+':00', regex=True)
    h = h.str.replace(r'^\d{2}$', lambda m: m.group(0)+':00', regex=True)
    h = h.str.replace(r'^(\d{1}):(\d{2})$', lambda m: m.group(1).zfill(2)+':'+m.group(2), regex=True)
    df[hora_col] = h
    ts = pd.to_datetime(df[data_col] + ' ' + df[hora_col], errors='coerce', utc=True)
    invalid_ts = ts.isna().sum()
    logging.info(f"Timestamps gerados. Inválidos/NaT: {invalid_ts:,} de {len(ts):,}")
    df.insert(0, 'timestamp', ts)
    num_candidates = [
        'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)',
        'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)',
        'PRESSÃO ATMOSFERICA MAX.NA HORA ANT. (AUT) (mB)',
        'PRESSÃO ATMOSFERICA MIN. NA HORA ANT. (AUT) (mB)',
        'RADIACAO GLOBAL (KJ/m²)',
        'RADIACAO GLOBAL (Kj/m²)',
        'TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)',
        'TEMPERATURA DO PONTO DE ORVALHO (°C)',
        'TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)',
        'TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)',
        'TEMPERATURA ORVALHO MAX. NA HORA ANT. (AUT) (°C)',
        'TEMPERATURA ORVALHO MIN. NA HORA ANT. (AUT) (°C)',
        'UMIDADE REL. MAX. NA HORA ANT. (AUT) (%)',
        'UMIDADE REL. MIN. NA HORA ANT. (AUT) (%)',
        'UMIDADE RELATIVA DO AR, HORARIA (%)',
        'VENTO, DIREÇÃO HORARIA (gr) (° (gr))',
        'VENTO, RAJADA MAXIMA (m/s)',
        'VENTO, VELOCIDADE HORARIA (m/s)'
    ]
    found = 0
    for c in num_candidates:
        if c in df.columns:
            df[c] = coerce_numeric(df[c], c); found += 1
    logging.info(f"Colunas numéricas convertidas: {found}/{len(num_candidates)}")
    ur_col = 'UMIDADE RELATIVA DO AR, HORARIA (%)'
    if ur_col in df.columns:
        before = df[ur_col].isna().sum()
        df.loc[(df[ur_col] < 0) | (df[ur_col] > 100), ur_col] = np.nan
        after = df[ur_col].isna().sum()
        logging.info(f"UR sanitizada: NaN antes={before:,} → depois={after:,}")
    _str_cols = {'timestamp', data_col, hora_col, '_file', '_sheet', 'Unnamed: 19'}
    for col in df.select_dtypes(include='object').columns:
        if col not in _str_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    before_rows = len(df)
    df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    logging.info(f"Linhas com timestamp válido: {len(df):,} (descartadas {before_rows-len(df):,})")
    return df
def build_qc_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['year'] = pd.to_datetime(df['timestamp']).dt.year
    cols = [c for c in df.columns if c not in ['timestamp', 'year']]
    records = []
    logging.info("Computando QC por ano×coluna...")
    for y, g in df.groupby('year', sort=True):
        rows = len(g)
        for c in cols:
            non_null = int(g[c].notna().sum())
            miss = 100.0 * (1.0 - non_null / max(rows, 1))
            records.append({'year': int(y), 'column': c, 'rows': rows, 'non_null': non_null, 'missing_pct': round(miss, 3)})
    qc = pd.DataFrame.from_records(records).sort_values(['year','column'])
    logging.info(f"QC pronto: {len(qc):,} linhas")
    return qc
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
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw-dir', default='data/raw')
    ap.add_argument('--out-dir', default='data/ints')
    ap.add_argument('--log-level', default='INFO', help='DEBUG, INFO, WARNING, ERROR')
    args = ap.parse_args()
    setup_logger(args.log_level)
    raw_dir=Path(args.raw_dir); out_dir=Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"=== INÍCIO | raw={raw_dir} -> out={out_dir} ===")
    df = clean_and_unify_xlsx(raw_dir)
    pq = out_dir/'inmet_sp_hourly_clean.parquet'
    logging.info(f"Salvando Parquet em {pq} ...")
    df.to_parquet(pq, index=False)
    logging.info(f"Parquet salvo ({human_size(pq)}).")
    qc = build_qc_table(df); qc_path = out_dir/'qc_hourly.csv'
    logging.info(f"Salvando QC em {qc_path} ...")
    qc.to_csv(qc_path, index=False, encoding='utf-8')
    logging.info("=== FIM | OK ===")
if __name__=='__main__':
    main()
