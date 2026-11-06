# -*- coding: utf-8 -*-
"""run_all.py: confere as entradas antes de cada etapa e para com mensagem clara (o professor
rodou o passo 3 sem legado_v1/data/ e recebeu um FileNotFoundError cru no Exp 0)."""
import importlib.util
from pathlib import Path

import pytest

AQUI = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("run_all", AQUI / "run_all.py")
ra = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ra)

TODAS = ra.ETAPAS_COLETA + ra.ETAPAS_BASE + ra.ETAPAS_CONGELAR + ra.ETAPAS_TESTE


def _cria(raiz, script):
    for caminho, _ in ra.ENTRADAS[script]:
        p = raiz / caminho.replace("*", "20260101")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x", encoding="utf-8")


def test_toda_etapa_tem_entradas_declaradas():
    for _, cmd in TODAS:
        assert cmd[0] in ra.ENTRADAS, cmd[0]
        assert (AQUI / cmd[0]).is_file(), cmd[0]


def test_run_all_padrao_nao_congela_nem_abre_teste():
    scripts = [cmd[0] for _, cmd in ra.ETAPAS_BASE]
    assert "03_congelar_e_testar.py" not in scripts
    assert "03b_teste_diagnostico.py" not in scripts


def test_exp0_sem_legado_para_com_caminho_e_origem(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        ra.confere_entradas("Exp 0", "exp0_diagnostico_v1.py", raiz=tmp_path)
    assert e.value.code == 2
    out = capsys.readouterr().out
    assert "legado_v1/data/features/inmet_sp_daily.parquet" in out
    assert "legado_v1/README.md" in out


def test_entradas_presentes_nao_interrompem(tmp_path):
    for _, cmd in TODAS:
        _cria(tmp_path, cmd[0])
    for rotulo, cmd in TODAS:
        ra.confere_entradas(rotulo, cmd[0], raiz=tmp_path)


def test_glob_exige_um_arquivo_por_horizonte(tmp_path):
    _cria(tmp_path, "exp0_diagnostico_v1.py")
    next((tmp_path / "legado_v1/reports/threshold_otimizado").glob("*y60*")).unlink()
    faltam = [c for c, _ in ra.entradas_faltando("exp0_diagnostico_v1.py", raiz=tmp_path)]
    assert faltam == ["legado_v1/reports/threshold_otimizado/comparativo_modelos_y60_*.csv"]


def test_pre_checagem_so_cobra_entradas_externas(tmp_path, capsys):
    # sem dados brutos: a pre-checagem para; o dataset (gerado no proprio run) nao e cobrado
    with pytest.raises(SystemExit):
        ra.pre_checagem(ra.ETAPAS_BASE, coleta=False, raiz=tmp_path)
    out = capsys.readouterr().out
    assert "data/raw_v2/sar_cantareira_sistema_diario.csv" in out
    assert "dataset_diario.parquet" not in out
    # com --coleta os brutos vem da propria coleta; so a v1 continua sendo cobrada
    with pytest.raises(SystemExit):
        ra.pre_checagem(ra.ETAPAS_COLETA + ra.ETAPAS_BASE, coleta=True, raiz=tmp_path)
    out = capsys.readouterr().out
    assert "data/raw_v2/" not in out and "legado_v1/data/" in out
