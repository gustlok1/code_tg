#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
00_download_dados.py — Coleta de dados REAIS para o pipeline v2 (reestruturação do TG).
Etapa 0 do pipeline_v2.

Baixa, com cache local em data/raw_v2/:

  SAR / ANA (padrão)  — série histórica DIÁRIA dos 4 reservatórios do Sistema Cantareira
                        (cota, volume útil hm³/%, afluência, defluência) desde ANO_INICIAL.
                        Fonte: https://www.ana.gov.br/sar0/MedicaoCantareira (tabela HTML; lxml).
  Open-Meteo (--openmeteo) — reanálise ERA5-Land DIÁRIA nos 4 pontos da bacia.
                        Fonte: https://archive-api.open-meteo.com
  NASA POWER (--power)      — reanálise DIÁRIA nos 4 pontos da bacia.
                        Fonte: https://power.larc.nasa.gov

Clima é baixado em formato LONGO (uma linha por ponto por dia); o 01_build_dataset_v2.py
faz a média da bacia entre os pontos.

Saídas (data/raw_v2/):
  sar_cantareira_diario.csv, sar_cantareira_sistema_diario.csv,
  openmeteo_pontos_diario.csv, nasapower_pontos_diario.csv, _manifest.json, cache/

Uso:
  python pipeline_v2/00_download_dados.py                 # SAR
  python pipeline_v2/00_download_dados.py --power         # NASA POWER
  python pipeline_v2/00_download_dados.py --openmeteo     # Open-Meteo (pode ir p/ 2o plano)
  python pipeline_v2/00_download_dados.py --all
  python pipeline_v2/00_download_dados.py --refresh --start 1984-01-01 --end 2024-12-31

Dependências: pip install pandas lxml pyyaml (urllib é da stdlib).
"""

import argparse
import hashlib
import io
import json
import logging
import sys
import time
import urllib.parse
import urllib.request
from datetime import date, datetime
from pathlib import Path

import pandas as pd

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

ROOT = Path(__file__).resolve().parents[1]
RAW_V2 = ROOT / "data" / "raw_v2"
CACHE = RAW_V2 / "cache"
MANIFEST = RAW_V2 / "_manifest.json"
CONFIG = Path(__file__).resolve().parent / "config.yaml"

# ----------------------------------------------------------------------------
# Config / metadados
# ----------------------------------------------------------------------------
def carregar_config() -> dict:
    if yaml is not None and CONFIG.exists():
        return yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
    return {}

CFG = carregar_config()
ANO_INICIAL = int(CFG.get("ano_inicial", 1984))

SAR_BASE = "https://www.ana.gov.br/sar0/MedicaoCantareira"

# nome -> (codigo_sar, lat, lon, municipio, capacidade_hm3_ref)
RESERVATORIOS = {
    "JAGUARI-JACAREI": (29001, -22.9246, -46.4270, "VARGEM/SP", 808.04),
    "CACHOEIRA":       (29002, -23.0511, -46.3200, "PIRACAIA/SP", 69.65),
    "ATIBAINHA":       (29003, -23.1757, -46.3936, "NAZARE PAULISTA/SP", 96.26),
    "PAIVA CASTRO":    (29004, -23.3300, -46.6795, "FRANCO DA ROCHA/SP", 7.61),
}

# pontos de clima (do config, senão os 4 reservatórios)
if CFG.get("pontos"):
    PONTOS = [(p["nome"], p["lat"], p["lon"]) for p in CFG["pontos"]]
else:
    PONTOS = [(n, v[1], v[2]) for n, v in RESERVATORIOS.items()]

CENTROIDE_LAT = round(sum(p[1] for p in PONTOS) / len(PONTOS), 4)
CENTROIDE_LON = round(sum(p[2] for p in PONTOS) / len(PONTOS), 4)

OPENMETEO_BASE = "https://archive-api.open-meteo.com/v1/archive"
# nome Open-Meteo -> nome neutro de saída
OPENMETEO_DAILY = {
    "precipitation_sum": "precip_mm",
    "et0_fao_evapotranspiration": "et0_mm",
    "temperature_2m_mean": "tmean_c",
    "temperature_2m_max": "tmax_c",
    "temperature_2m_min": "tmin_c",
    "shortwave_radiation_sum": "rad_mj",
}

POWER_BASE = "https://power.larc.nasa.gov/api/temporal/daily/point"
# parâmetro POWER -> nome neutro de saída
POWER_PARAMS = {
    "PRECTOTCORR": "precip_mm",
    "T2M": "tmean_c",
    "T2M_MAX": "tmax_c",
    "T2M_MIN": "tmin_c",
    "ALLSKY_SFC_SW_DWN": "rad_mj",
    "RH2M": "rh_pct",
    "WS2M": "wind_ms",
    "PS": "pressure_kpa",
}

USER_AGENT = "TG-Cantareira/2.0 (coleta academica; pipeline_v2)"
SAR_COLS = ["codigo", "reservatorio", "cota_m", "vol_util_hm3",
            "vol_util_pct", "afluencia_m3s", "defluencia_m3s", "data"]


# ----------------------------------------------------------------------------
# utilidades
# ----------------------------------------------------------------------------
def _forcar_utf8():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass


def setup_logger(level: str):
    _forcar_utf8()
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=lvl, format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S")


def http_get(url: str, timeout: int = 120, tentativas: int = 6) -> str:
    import urllib.error
    ultimo = None
    for i in range(1, tentativas + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            ultimo = e
            # 429/5xx são transitórios: espera mais (rate limit reseta por minuto)
            espera = 60 if e.code == 429 else 5 * i
            logging.warning(f"  tentativa {i}/{tentativas} HTTP {e.code}; aguardando {espera}s")
            time.sleep(espera)
        except Exception as e:  # noqa: BLE001
            ultimo = e
            logging.warning(f"  tentativa {i}/{tentativas} falhou: {e}")
            time.sleep(5 * i)
    raise RuntimeError(f"GET falhou após {tentativas} tentativas: {url} ({ultimo})")


def cache_get(nome: str, url: str, refresh: bool) -> str:
    CACHE.mkdir(parents=True, exist_ok=True)
    fp = CACHE / nome
    if fp.exists() and not refresh:
        logging.info(f"  [cache] {nome}")
        return fp.read_text(encoding="utf-8")
    logging.info(f"  [baixando] {nome}")
    conteudo = http_get(url)
    fp.write_text(conteudo, encoding="utf-8")
    time.sleep(0.4)
    return conteudo


def limpar_num(serie: pd.Series) -> pd.Series:
    """float robusto: passa adiante se já numérico; senão parseia pt-BR ('1.234,56')."""
    if pd.api.types.is_numeric_dtype(serie):
        return serie.astype(float)
    s = serie.astype(str).str.strip()
    s = s.replace({"-": None, "": None, "nan": None, "None": None})
    s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def sha256_arquivo(fp: Path) -> str:
    h = hashlib.sha256(); h.update(fp.read_bytes()); return h.hexdigest()


def carregar_manifest() -> dict:
    if MANIFEST.exists():
        try:
            return json.loads(MANIFEST.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            logging.warning("Manifesto existente ilegível; recriando.")
    return {}


def salvar_manifest(man: dict):
    man["gerado_em"] = datetime.now().isoformat(timespec="seconds")
    man.setdefault("projeto", "TG Cantareira — pipeline_v2 (coleta de dados reais)")
    man["ano_inicial"] = ANO_INICIAL
    man["regiao"] = {
        "sistema": "Cantareira (bacias PCJ)",
        "centroide": {"lat": CENTROIDE_LAT, "lon": CENTROIDE_LON},
        "pontos": [{"nome": n, "lat": la, "lon": lo} for n, la, lo in PONTOS],
        "reservatorios": [
            {"nome": n, "codigo_sar": v[0], "lat": v[1], "lon": v[2],
             "municipio": v[3], "capacidade_hm3_ref": v[4]}
            for n, v in RESERVATORIOS.items()
        ],
    }
    MANIFEST.write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")


# ----------------------------------------------------------------------------
# SAR / ANA — Sistema Cantareira
# ----------------------------------------------------------------------------
def _parse_tabela_sar(html: str) -> pd.DataFrame:
    tabelas = pd.read_html(io.StringIO(html), match="Volume", flavor="lxml",
                           decimal=",", thousands=".")
    if not tabelas:
        raise ValueError("nenhuma tabela com 'Volume' encontrada")
    df = tabelas[0]
    if df.shape[1] != 8:
        raise ValueError(
            f"esperadas 8 colunas, vieram {df.shape[1]}: {list(df.columns)} "
            "(a estrutura da página do SAR mudou — ajuste SAR_COLS/_parse_tabela_sar)"
        )
    df.columns = SAR_COLS
    df = df[df["codigo"].astype(str).str.strip().str.isdigit()].copy()
    return df


def baixar_sar(args) -> dict:
    logging.info("=== SAR / ANA — Sistema Cantareira ===")
    d_ini = datetime.fromisoformat(args.start).date()
    d_fim = datetime.fromisoformat(args.end).date()
    anos = list(range(d_ini.year, d_fim.year + 1))

    partes, cache_files, falhas = [], [], []
    for nome, (codigo, *_rest) in RESERVATORIOS.items():
        logging.info(f"- {nome} (cod {codigo})")
        for ano in anos:
            a_ini = max(date(ano, 1, 1), d_ini).strftime("%d/%m/%Y")
            a_fim = min(date(ano, 12, 31), d_fim).strftime("%d/%m/%Y")
            qs = urllib.parse.urlencode({
                "dropDownListReservatorios": codigo,
                "dataInicial": a_ini, "dataFinal": a_fim, "auth": "False",
            })
            nome_cache = f"sar_{codigo}_{ano}.html"
            try:
                html = cache_get(nome_cache, f"{SAR_BASE}?{qs}", args.refresh)
                cache_files.append(nome_cache)
                df = _parse_tabela_sar(html)
                if len(df):
                    partes.append(df)
            except ValueError as e:
                logging.debug(f"    {ano}: sem dados ({e})")
            except Exception as e:  # noqa: BLE001
                logging.error(f"    {ano}: FALHA {e}")
                falhas.append({"reservatorio": nome, "ano": ano, "erro": str(e)})

    if not partes:
        raise RuntimeError(
            "Nenhum dado do SAR obtido. Abra uma URL de exemplo e ajuste _parse_tabela_sar():\n  "
            f"{SAR_BASE}?dropDownListReservatorios=29001&dataInicial=01/01/1984&"
            "dataFinal=10/01/1984&auth=False"
        )

    tidy = pd.concat(partes, ignore_index=True)
    tidy["codigo"] = pd.to_numeric(tidy["codigo"], errors="coerce").astype("Int64")
    tidy["reservatorio"] = tidy["reservatorio"].astype(str).str.strip()
    for c in ["cota_m", "vol_util_hm3", "vol_util_pct", "afluencia_m3s", "defluencia_m3s"]:
        tidy[c] = limpar_num(tidy[c])
    tidy["data"] = pd.to_datetime(tidy["data"], format="%d/%m/%Y", errors="coerce")
    tidy = (tidy.dropna(subset=["data"])
                .drop_duplicates(subset=["codigo", "data"])
                .sort_values(["data", "codigo"]).reset_index(drop=True))
    tidy = tidy[["data", "codigo", "reservatorio", "cota_m", "vol_util_hm3",
                 "vol_util_pct", "afluencia_m3s", "defluencia_m3s"]]

    out_tidy = RAW_V2 / "sar_cantareira_diario.csv"
    tidy.to_csv(out_tidy, index=False, encoding="utf-8", date_format="%Y-%m-%d",
                float_format="%.4f")
    sistema = _agregar_sistema(tidy)
    out_sis = RAW_V2 / "sar_cantareira_sistema_diario.csv"
    sistema.to_csv(out_sis, index=False, encoding="utf-8", date_format="%Y-%m-%d",
                   float_format="%.4f")
    logging.info(f"SAR tidy: {len(tidy):,} linhas -> {out_tidy.name}")
    logging.info(f"SAR sistema: {len(sistema):,} dias -> {out_sis.name}")

    return {
        "fonte": "ANA/SAR — Medição Cantareira",
        "url_base": SAR_BASE,
        "reservatorios": {n: v[0] for n, v in RESERVATORIOS.items()},
        "periodo_solicitado": [args.start, args.end],
        "periodo_obtido": [tidy["data"].min().strftime("%Y-%m-%d"),
                           tidy["data"].max().strftime("%Y-%m-%d")],
        "linhas_tidy": int(len(tidy)), "dias_sistema": int(len(sistema)),
        "arquivos": {"diario": out_tidy.name, "sistema": out_sis.name},
        "sha256": {out_tidy.name: sha256_arquivo(out_tidy),
                   out_sis.name: sha256_arquivo(out_sis)},
        "cache_dir": "cache/", "n_arquivos_cache": len(cache_files), "falhas": falhas,
        "atualizado_em": datetime.now().isoformat(timespec="seconds"),
    }


def _agregar_sistema(tidy: pd.DataFrame) -> pd.DataFrame:
    """Agrega os reservatórios no volume diário do sistema. NÃO corta valores negativos:
    o volume útil pode ser < 0 (reserva técnica/volume morto), como em 2014-2015."""
    cap = {}
    for cod, g in tidy.groupby("codigo"):
        gg = g[g["vol_util_pct"] > 1]          # só p/ estimar capacidade de forma estável
        cap[cod] = (gg["vol_util_hm3"] / (gg["vol_util_pct"] / 100.0)).median()
    tidy = tidy.copy()
    tidy["cap_hm3"] = tidy["codigo"].map(cap)

    def _por_dia(g: pd.DataFrame) -> pd.Series:
        vol = g["vol_util_hm3"].sum(min_count=1)        # pode ser negativo (real)
        capacidade = g.loc[g["vol_util_hm3"].notna(), "cap_hm3"].sum(min_count=1)
        pct = 100.0 * vol / capacidade if capacidade and capacidade > 0 else float("nan")
        return pd.Series({
            "vol_util_hm3_sistema": vol,
            "capacidade_hm3_sistema": capacidade,
            "vol_util_pct_sistema": pct,
            "afluencia_m3s_sistema": g["afluencia_m3s"].sum(min_count=1),
            "defluencia_m3s_sistema": g["defluencia_m3s"].sum(min_count=1),
            "n_reservatorios": int(g["vol_util_hm3"].notna().sum()),
        })

    sis = tidy.groupby("data").apply(_por_dia, include_groups=False)
    sis = sis.reset_index().sort_values("data").reset_index(drop=True)
    sis["n_reservatorios"] = sis["n_reservatorios"].astype(int)
    return sis


# ----------------------------------------------------------------------------
# Open-Meteo (ERA5-Land) — multiponto
# ----------------------------------------------------------------------------
def baixar_openmeteo(args) -> dict:
    logging.info("=== Open-Meteo (ERA5) — 4 pontos da bacia ===")
    d_fim = min(datetime.fromisoformat(args.end).date(), date.today())
    partes = []
    for nome, lat, lon in PONTOS:
        qs = urllib.parse.urlencode({
            "latitude": lat, "longitude": lon,
            "start_date": args.start, "end_date": d_fim.isoformat(),
            "daily": ",".join(OPENMETEO_DAILY.keys()),
            "timezone": "America/Sao_Paulo",
        })
        raw = cache_get(f"era5_{nome}_{args.start}_{d_fim.isoformat()}.json".replace(" ", "_"),
                        f"{OPENMETEO_BASE}?{qs}", args.refresh)
        data = json.loads(raw)
        if "daily" not in data:
            raise RuntimeError(f"Open-Meteo sem 'daily' em {nome}: {str(data)[:200]}")
        df = pd.DataFrame(data["daily"]).rename(columns={"time": "data", **OPENMETEO_DAILY})
        df.insert(0, "ponto", nome); df["lat"] = lat; df["lon"] = lon
        partes.append(df)
    longo = pd.concat(partes, ignore_index=True)
    longo["data"] = pd.to_datetime(longo["data"])
    longo = longo.sort_values(["data", "ponto"]).reset_index(drop=True)
    cols = ["data", "ponto", "lat", "lon"] + list(OPENMETEO_DAILY.values())
    longo = longo[cols]
    out = RAW_V2 / "openmeteo_pontos_diario.csv"
    longo.to_csv(out, index=False, encoding="utf-8", date_format="%Y-%m-%d")
    logging.info(f"Open-Meteo: {len(longo):,} linhas ({longo['ponto'].nunique()} pontos) -> {out.name}")
    return {
        "fonte": "Open-Meteo Historical Weather API (ERA5-Land)",
        "url_base": OPENMETEO_BASE,
        "pontos": [p[0] for p in PONTOS],
        "variaveis": list(OPENMETEO_DAILY.values()),
        "periodo_solicitado": [args.start, args.end],
        "periodo_obtido": [longo["data"].min().strftime("%Y-%m-%d"),
                           longo["data"].max().strftime("%Y-%m-%d")],
        "linhas": int(len(longo)), "arquivo": out.name,
        "sha256": {out.name: sha256_arquivo(out)},
        "atualizado_em": datetime.now().isoformat(timespec="seconds"),
    }


# ----------------------------------------------------------------------------
# NASA POWER — multiponto
# ----------------------------------------------------------------------------
def baixar_power(args) -> dict:
    logging.info("=== NASA POWER — 4 pontos da bacia ===")
    d_ini = datetime.fromisoformat(args.start).date()
    d_fim = min(datetime.fromisoformat(args.end).date(), date.today())
    partes = []
    for nome, lat, lon in PONTOS:
        qs = urllib.parse.urlencode({
            "parameters": ",".join(POWER_PARAMS.keys()), "community": "AG",
            "latitude": lat, "longitude": lon,
            "start": d_ini.strftime("%Y%m%d"), "end": d_fim.strftime("%Y%m%d"),
            "format": "JSON",
        })
        raw = cache_get(f"power_{nome}_{args.start}_{d_fim.isoformat()}.json".replace(" ", "_"),
                        f"{POWER_BASE}?{qs}", args.refresh)
        data = json.loads(raw)
        try:
            par = data["properties"]["parameter"]
        except KeyError:
            raise RuntimeError(f"NASA POWER sem 'parameter' em {nome}: {str(data)[:200]}")
        df = pd.DataFrame({saida: pd.Series(par[p]) for p, saida in POWER_PARAMS.items()})
        df.index.name = "data"; df = df.reset_index()
        df["data"] = pd.to_datetime(df["data"], format="%Y%m%d")
        df = df.replace(-999.0, pd.NA)          # sentinela de ausente do POWER
        df.insert(1, "ponto", nome); df["lat"] = lat; df["lon"] = lon
        partes.append(df)
    longo = pd.concat(partes, ignore_index=True).sort_values(["data", "ponto"]).reset_index(drop=True)
    cols = ["data", "ponto", "lat", "lon"] + list(POWER_PARAMS.values())
    longo = longo[cols]
    out = RAW_V2 / "nasapower_pontos_diario.csv"
    longo.to_csv(out, index=False, encoding="utf-8", date_format="%Y-%m-%d")
    logging.info(f"NASA POWER: {len(longo):,} linhas ({longo['ponto'].nunique()} pontos) -> {out.name}")
    return {
        "fonte": "NASA POWER (temporal/daily/point, community=AG)",
        "url_base": POWER_BASE,
        "pontos": [p[0] for p in PONTOS],
        "variaveis": list(POWER_PARAMS.values()),
        "periodo_solicitado": [args.start, args.end],
        "periodo_obtido": [longo["data"].min().strftime("%Y-%m-%d"),
                           longo["data"].max().strftime("%Y-%m-%d")],
        "linhas": int(len(longo)), "arquivo": out.name,
        "sha256": {out.name: sha256_arquivo(out)},
        "atualizado_em": datetime.now().isoformat(timespec="seconds"),
    }


# ----------------------------------------------------------------------------
def imprimir_resumo(man: dict):
    print("\n" + "=" * 70)
    print("RESUMO DA COLETA — pipeline_v2 (data/raw_v2/)")
    print("=" * 70)
    print(f"Gerado em   : {man.get('gerado_em')}")
    print(f"Ano inicial : {man.get('ano_inicial')}")
    print(f"Região      : Sistema Cantareira  (centroide {CENTROIDE_LAT}, {CENTROIDE_LON})")
    f = man.get("fontes", {})

    s = f.get("sar_cantareira")
    if s:
        print("\n[SAR / ANA — Sistema Cantareira]")
        print(f"  Reservatórios  : {', '.join(s['reservatorios'])}")
        print(f"  Período obtido : {s['periodo_obtido'][0]} -> {s['periodo_obtido'][1]}")
        print(f"  Linhas (tidy)  : {s['linhas_tidy']:,}  |  dias (sistema): {s['dias_sistema']:,}")
        print(f"  Cache          : {s['n_arquivos_cache']} arquivos")
        if s.get("falhas"):
            print(f"  ATENÇÃO        : {len(s['falhas'])} falha(s) parcial(is) — ver _manifest.json")

    o = f.get("openmeteo")
    if o:
        print("\n[Open-Meteo (ERA5) — pontos da bacia]")
        print(f"  Período obtido : {o['periodo_obtido'][0]} -> {o['periodo_obtido'][1]}")
        print(f"  Linhas         : {o.get('linhas', 0):,}  "
              f"({len(o.get('pontos', []))} pontos)  -> {o.get('arquivo', '?')}")

    p = f.get("nasapower")
    if p:
        print("\n[NASA POWER — pontos da bacia]")
        print(f"  Período obtido : {p['periodo_obtido'][0]} -> {p['periodo_obtido'][1]}")
        print(f"  Linhas         : {p.get('linhas', 0):,}  "
              f"({len(p.get('pontos', []))} pontos)  -> {p.get('arquivo', '?')}")

    print(f"\nManifesto   : {MANIFEST}")
    print("=" * 70)


def main():
    ap = argparse.ArgumentParser(description="Coleta de dados reais do pipeline_v2.")
    ap.add_argument("--openmeteo", action="store_true", help="Baixa o Open-Meteo (ERA5).")
    ap.add_argument("--power", action="store_true", help="Baixa o NASA POWER.")
    ap.add_argument("--sar", action="store_true", help="Força a coleta do SAR.")
    ap.add_argument("--all", action="store_true", help="Baixa SAR + Open-Meteo + NASA POWER.")
    ap.add_argument("--start", default=f"{ANO_INICIAL}-01-01", help="Data inicial (YYYY-MM-DD).")
    ap.add_argument("--end", default=date.today().isoformat(), help="Data final (YYYY-MM-DD).")
    ap.add_argument("--refresh", action="store_true", help="Ignora o cache e rebaixa.")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    setup_logger(args.log_level)
    RAW_V2.mkdir(parents=True, exist_ok=True)

    algum_flag = args.openmeteo or args.power or args.sar or args.all
    fazer_sar = args.all or args.sar or not algum_flag      # padrão = SAR
    fazer_om = args.all or args.openmeteo
    fazer_pw = args.all or args.power

    logging.info(f"=== INÍCIO | SAR={fazer_sar} ERA5={fazer_om} POWER={fazer_pw} | "
                 f"{args.start} -> {args.end} ===")
    man = carregar_manifest(); man.setdefault("fontes", {})

    if fazer_sar:
        man["fontes"]["sar_cantareira"] = baixar_sar(args); salvar_manifest(man)
    if fazer_om:
        man["fontes"]["openmeteo"] = baixar_openmeteo(args); salvar_manifest(man)
    if fazer_pw:
        man["fontes"]["nasapower"] = baixar_power(args); salvar_manifest(man)

    salvar_manifest(man)
    imprimir_resumo(man)
    logging.info("=== FIM coleta | OK ===")


if __name__ == "__main__":
    main()
