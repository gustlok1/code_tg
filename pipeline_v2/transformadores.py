# -*- coding: utf-8 -*-
"""
transformadores.py — Indicadores padronizados (SPI e SPEI) com ajuste gama.

Estes indicadores dependem de ESTATÍSTICAS DA SÉRIE (climatologia mensal), portanto
NÃO podem entrar em 01_build_dataset_v2.py (que só produz features causais). Aqui eles
seguem a API fit/transform do scikit-learn:

    spi = SPI(escala_meses=3).fit(serie_mensal_TREINO)
    z_treino = spi.transform(serie_mensal_TREINO)
    z_teste  = spi.transform(serie_mensal_QUALQUER)

Assim o ajuste (parâmetros da gama por mês-do-ano) é feito SÓ no treino, dentro de
cada fold, e aplicado a qualquer período — sem vazamento.

- SPI  (Standardized Precipitation Index): sobre a chuva acumulada em k meses.
- SPEI (Standardized Precipitation-Evapotranspiration Index): sobre o balanço
  hídrico D = P − ET0 acumulado em k meses. O SPEI clássico usa log-logística;
  aqui, conforme especificado, usa-se ajuste GAMA sobre D deslocado para o
  domínio positivo (offset estimado no treino).

Referência: McKee et al. (1993) para o SPI; Vicente-Serrano et al. (2010) para o SPEI.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import gamma, norm

_EPS = 1e-6


def serie_mensal(df: pd.DataFrame, col_data: str, col_valor: str,
                 como: str = "soma") -> pd.Series:
    """Agrega uma coluna diária para mensal (soma p/ fluxos; média p/ estados).
    Retorna Series indexada pelo fim do mês."""
    s = (df[[col_data, col_valor]].dropna(subset=[col_data]).copy())
    s[col_data] = pd.to_datetime(s[col_data])
    s = s.set_index(col_data)[col_valor]
    regra = {"soma": "sum", "media": "mean"}[como]
    return s.resample("ME").agg(regra)


class _GamaPorMes:
    """Padroniza uma série de acumulações mensais via gama ajustada por mês-do-ano.

    Para cada mês (1..12) guarda (q, shape, scale), onde q é a fração de valores
    nulos (chuva zero) tratada como massa pontual: H(x) = q + (1−q)·G(x).
    """

    def __init__(self, escala_meses: int):
        if escala_meses < 1:
            raise ValueError("escala_meses deve ser >= 1")
        self.escala = int(escala_meses)
        self.params_: dict[int, tuple[float, float, float]] = {}
        self.offset_: float = 0.0          # deslocamento (usado pelo SPEI)
        self.ajustado_ = False

    # --- a acumulação é definida pela subclasse (chuva ou balanço) ---
    def _acumular(self, mensal: pd.Series) -> pd.Series:
        return mensal.rolling(self.escala, min_periods=self.escala).sum()

    def fit(self, mensal: pd.Series) -> "_GamaPorMes":
        acc = (self._acumular(mensal) + self.offset_).dropna()
        for m in range(1, 13):
            x = acc[acc.index.month == m].values.astype(float)
            if x.size == 0:
                self.params_[m] = (np.nan, np.nan, np.nan)
                continue
            q = float((x <= 0).mean())
            pos = x[x > 0]
            if pos.size >= 4 and np.ptp(pos) > 0:
                shape, _loc, scale = gamma.fit(pos, floc=0)
            else:
                shape, scale = np.nan, np.nan
            self.params_[m] = (q, float(shape), float(scale))
        self.ajustado_ = True
        return self

    def transform(self, mensal: pd.Series) -> pd.Series:
        if not self.ajustado_:
            raise RuntimeError("Chame fit() no treino antes de transform().")
        acc = self._acumular(mensal) + self.offset_
        out = pd.Series(np.nan, index=acc.index, dtype=float)
        for ts, x in acc.items():
            if pd.isna(x):
                continue
            q, shape, scale = self.params_.get(ts.month, (np.nan, np.nan, np.nan))
            if not np.isfinite(shape) or not np.isfinite(scale):
                continue
            if x <= 0:
                H = q
            else:
                H = q + (1.0 - q) * float(gamma.cdf(x, shape, loc=0, scale=scale))
            H = min(max(H, _EPS), 1.0 - _EPS)
            out[ts] = float(norm.ppf(H))
        return out

    def fit_transform(self, mensal: pd.Series) -> pd.Series:
        return self.fit(mensal).transform(mensal)


class SPI(_GamaPorMes):
    """SPI: padroniza a chuva mensal acumulada em `escala_meses`."""
    # usa _acumular herdado (soma da chuva); offset_ = 0


class SPEI(_GamaPorMes):
    """SPEI: padroniza o balanço D = P − ET0 acumulado em `escala_meses`.

    Como D pode ser negativo e a gama exige suporte positivo, o fit estima um
    `offset_` (no treino) que desloca as acumulações para o domínio positivo;
    o mesmo offset é reaplicado no transform.
    """

    def fit(self, mensal_balanco: pd.Series) -> "SPEI":
        acc = mensal_balanco.rolling(self.escala, min_periods=self.escala).sum().dropna()
        minimo = float(acc.min()) if len(acc) else 0.0
        self.offset_ = (-minimo + 1.0) if minimo <= 0 else 0.0
        return super().fit(mensal_balanco)


def padronizar_para_diario(df_diario: pd.DataFrame, col_data: str,
                           z_mensal: pd.Series, nome: str) -> pd.Series:
    """Projeta um índice mensal (z) de volta para o calendário diário: cada dia recebe
    o valor do seu mês. Útil para anexar SPI/SPEI ao dataset diário DENTRO do fold."""
    d = pd.to_datetime(df_diario[col_data])
    chave_dia = d.dt.to_period("M")
    chave_z = z_mensal.copy()
    chave_z.index = z_mensal.index.to_period("M")
    return pd.Series(chave_dia.map(chave_z).values, index=df_diario.index, name=nome)


def aplicar_indicadores(df_diario: pd.DataFrame, mask_treino, *,
                        col_data="data", col_precip="precip_mm", col_et0="et0_mm",
                        escalas=(3, 6, 12)) -> pd.DataFrame:
    """Conveniência para uso em treino: ajusta SPI/SPEI SÓ nas linhas de treino e
    transforma a série inteira. `mask_treino` é booleano alinhado a df_diario.

    Retorna um DataFrame (mesmo índice de df_diario) com colunas spi_{k} e spei_{k}.
    """
    df = df_diario.copy()
    df[col_data] = pd.to_datetime(df[col_data])
    precip_m = serie_mensal(df, col_data, col_precip, "soma")
    tem_et0 = col_et0 in df.columns
    if tem_et0:
        df["_D"] = df[col_precip] - df[col_et0]
        bal_m = serie_mensal(df, col_data, "_D", "soma")

    # máscara mensal: um mês é "de treino" se a maioria dos seus dias é de treino
    mes_treino = (pd.Series(mask_treino.values, index=df[col_data])
                  .resample("ME").mean() >= 0.5)

    saida = pd.DataFrame(index=df.index)
    for k in escalas:
        spi = SPI(k).fit(precip_m[mes_treino.reindex(precip_m.index, fill_value=False)])
        z = spi.transform(precip_m)
        saida[f"spi_{k}"] = padronizar_para_diario(df, col_data, z, f"spi_{k}")
        if tem_et0:
            spei = SPEI(k).fit(bal_m[mes_treino.reindex(bal_m.index, fill_value=False)])
            zz = spei.transform(bal_m)
            saida[f"spei_{k}"] = padronizar_para_diario(df, col_data, zz, f"spei_{k}")
    return saida


# ============================================================================
# Anomalia sazonal (climatologia por dia do ano, ajustada SÓ no treino)
# ============================================================================
class ClimatologiaSazonal:
    """Climatologia de uma variável diária por dia-do-ano (1..365), ajustada no treino.

    fit(datas_treino, valores_treino) estima a média por dia-do-ano (29/02 -> 365);
    transform(datas, valores) devolve a ANOMALIA = valor − climatologia[dia-do-ano].
    Como o fit usa só o treino, alterar dados de validação não muda a climatologia.
    """

    def __init__(self):
        self.clim_: dict[int, float] = {}
        self.media_: float = np.nan
        self.ajustado_ = False

    @staticmethod
    def _doy(datas) -> np.ndarray:
        d = pd.to_datetime(datas).dt.dayofyear.values
        return np.minimum(d, 365)

    def fit(self, datas, valores) -> "ClimatologiaSazonal":
        doy = self._doy(pd.Series(pd.to_datetime(datas)))
        s = pd.Series(np.asarray(valores, dtype=float), index=doy).dropna()
        self.clim_ = s.groupby(level=0).mean().to_dict()
        self.media_ = float(np.nanmean(list(self.clim_.values()))) if self.clim_ else np.nan
        self.ajustado_ = True
        return self

    def transform(self, datas, valores) -> np.ndarray:
        if not self.ajustado_:
            raise RuntimeError("Chame fit() no treino antes de transform().")
        doy = self._doy(pd.Series(pd.to_datetime(datas)))
        clim = np.array([self.clim_.get(d, self.media_) for d in doy], dtype=float)
        return np.asarray(valores, dtype=float) - clim


def anomalias_sazonais(df: pd.DataFrame, mask_treino, cols, col_data: str = "data") -> pd.DataFrame:
    """Anomalia sazonal de cada coluna em `cols`, climatologia ajustada só no treino.
    Retorna DataFrame (índice de df) com colunas 'anom_<col>'."""
    mask = np.asarray(mask_treino)
    datas = pd.to_datetime(df[col_data])
    out = pd.DataFrame(index=df.index)
    for c in cols:
        cl = ClimatologiaSazonal().fit(datas[mask], df[c].values[mask])
        out[f"anom_{c}"] = cl.transform(datas, df[c].values)
    return out
