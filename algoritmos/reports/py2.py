
def formatarMoeda(valor):
    """Formata o valor para padrão brasileiro: R$ 1.234,56"""
    return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

# =========================
# PARÂMETROS PRINCIPAIS (CARTEIRA)
# =========================
anos = 5                                  # Período da análise (anos)
diasPorMes = 22                           # Dias úteis por mês
jornadaPadrao = 8                         # Jornada padrão (horas/dia)

numProcessosAutomatizados = 30 * anos     # Total de processos automatizados ao final do período

# Parâmetros médios por processo (Ano 1)
custoColaborador = 6000                   # Salário mensal por colaborador (R$)
demandaMensalInicial = 22                 # Execuções/mês por processo
casosPorExecucaoInicial = 1               # Casos por execução
colaboradoresIniciais = 1                 # Pessoas por processo (base)
horasManualPorDia = (60 / 60) * 1         # Tempo manual/dia por colaborador
horasRPAPorDia = (10 / 60)                # Tempo de robô/dia (referência de carga)

# Crescimentos médios (geométricos ao longo dos anos)
crescimentoVolume = 2                     # Fator total para execuções (ex.: 2.0 = dobrar)
crescimentoCasos = 1                      # Fator total para casos/execução

# Célula de RPA (mensal) — SEM DILUIÇÃO (mostra o todo gasto com RPA)
custoRPASalario = (5000 * 2)              # Salário analista RPA / mês
custoRPAVm = 6000                         # Custo VM / mês
custoRPALicenca = (800 * 2)               # Licença / mês
custoCelulaRPA_Mensal = custoRPASalario + custoRPAVm + custoRPALicenca

# Distribuição (rampa) de processos ativos por ano — cumulativa
# Se None, usa rampa linear cumulativa automática (ex.: 60/3 -> [20, 40, 60])
distribuicaoPorAno = None  # Exemplo: [20, 40, 60]

# =========================
# EXECUÇÃO
# =========================
def analisarCarteiraAnualComRampa_UsoPercent(
    anos=anos, diasPorMes=diasPorMes, jornadaPadrao=jornadaPadrao,
    numProcessosAutomatizados=numProcessosAutomatizados, distribuicaoPorAno=distribuicaoPorAno,
    custoColaborador=custoColaborador, demandaMensalInicial=demandaMensalInicial,
    casosPorExecucaoInicial=casosPorExecucaoInicial, colaboradoresIniciais=colaboradoresIniciais,
    horasManualPorDia=horasManualPorDia, horasRPAPorDia=horasRPAPorDia,
    crescimentoVolume=crescimentoVolume, crescimentoCasos=crescimentoCasos,
    custoCelulaRPA_Mensal=custoCelulaRPA_Mensal
):
    """
    Carteira anual com rampa de processos ativos e uso (%) proporcional ao volume:
      - Uso base (%) = horasManualPorDia / jornadaPadrao.
      - Uso colab. p/ proc. (%) = Uso base × (Carga relativa do ano).
      - Carga relativa = (execuções × casos/execução do ano) / (execuções × casos/execução do Ano 1).
      - Custos 'Sem RPA' por ano = processos ativos × custo colaborador × Uso colab. p/ proc. (%) × 12 × colaboradoresIniciais.
      - Custo RPA por ano = célula × 12 (sem diluição, custo total).
      - FTE, ROI e Payback em anos calculados para carteira ativa.
    """

    # Construção da rampa se não fornecida
    if not distribuicaoPorAno:
        dist = []
        for i in range(1, anos + 1):
            dist.append((numProcessosAutomatizados * i + anos - 1) // anos)  # ceil linear cumulativa
        dist = [min(x, numProcessosAutomatizados) for x in dist]
        for i in range(1, anos):
            dist[i] = max(dist[i], dist[i-1])
        dist[-1] = numProcessosAutomatizados
        distribuicao = dist
    else:
        distribuicao = distribuicaoPorAno[:]
        if len(distribuicao) != anos:
            raise ValueError("distribuicaoPorAno deve ter o mesmo comprimento de 'anos'.")
        for i in range(1, anos):
            if distribuicao[i] < distribuicao[i-1]:
                raise ValueError("A distribuição por ano deve ser cumulativa (não decrescente).")
        if distribuicao[-1] != numProcessosAutomatizados:
            raise ValueError("O último ano deve igualar 'numProcessosAutomatizados' (todos automatizados).")

    # Fatores geométricos por ano
    fatorVolumePorAno = (crescimentoVolume ** (1 / (anos - 1))) if anos > 1 else 1.0
    fatorCasosPorAno = (crescimentoCasos ** (1 / (anos - 1))) if anos > 1 else 1.0

    # Uso base (%) por processo
    usoBaseFrac = horasManualPorDia / jornadaPadrao  # fração (ex.: 0.10 = 10%)
    usoBasePercent = usoBaseFrac * 100

    # Base de carga (Ano 1)
    cargaBase = demandaMensalInicial * casosPorExecucaoInicial

    # Acumuladores totais
    custoTotalManual = 0.0
    custoTotalRPA = 0.0
    economiaTotal = 0.0
    horasManualTotal = 0.0
    horasRPATotal = 0.0
    fteEconomizadosTotal = 0.0

    # Payback anual/mensal
    economiaAcumulada = 0.0
    paybackAno = None

    resultados_anuais = []
    roi_anos_percentuais = []

    # Guardas para cálculo mensal
    meses_totais = anos * 12
    paybackMes = None

    # Estado anual evolutivo
    demandaMensal = demandaMensalInicial
    casosPorExecucaoAtual = casosPorExecucaoInicial

    for ano in range(1, anos + 1):
        if ano > 1:
            demandaMensal *= fatorVolumePorAno
            casosPorExecucaoAtual *= fatorCasosPorAno

        processosAtivos = distribuicao[ano - 1]  # cumulativo no ano

        # Carga relativa do ano vs Ano 1
        cargaAno = demandaMensal * casosPorExecucaoAtual
        cargaRelativa = (cargaAno / cargaBase) if cargaBase > 0 else 1.0

        # Uso colab. p/ proc. (%) por processo no ano (sem cap, conforme modelo original)
        usoAjustadoFrac = usoBaseFrac * cargaRelativa
        usoAjustadoPercent = usoAjustadoFrac * 100

        # Custos Sem RPA (ano) para processos ativos (refletindo colaboradoresIniciais)
        custoSemRPA_ano = processosAtivos * custoColaborador * usoAjustadoFrac * 12 * colaboradoresIniciais
        custoTotalManual += custoSemRPA_ano

        # Horas manuais (ano) — por processo base × cargaRelativa × processosAtivos
        horasManualPorMes_base_processo = horasManualPorDia * diasPorMes * colaboradoresIniciais
        horasManualAno = horasManualPorMes_base_processo * 12 * cargaRelativa * processosAtivos
        horasManualTotal += horasManualAno

        # Custo e horas de RPA (ano) — sem diluição: custo da célula * 12
        custoComRPA_ano = custoCelulaRPA_Mensal * 12
        horasRPA_ano = horasRPAPorDia * diasPorMes * 12 * processosAtivos  # referência de carga total
        horasRPATotal += horasRPA_ano

        # Economia (ano) e FTE
        economia_ano = custoSemRPA_ano - custoComRPA_ano
        economiaTotal += economia_ano

        horasPorColaboradorAno = jornadaPadrao * diasPorMes * 12
        fteEconomizadosAno = horasManualAno / horasPorColaboradorAno
        fteEconomizadosTotal += fteEconomizadosAno

        # Custo total de RPA
        custoTotalRPA += custoComRPA_ano

        # Payback (economia acumulada vs 0): break-even operacional (anual)
        economiaAcumulada += economia_ano
        if paybackAno is None and economiaAcumulada >= 0:
            paybackAno = ano

        # ROI por ano (economia do ano / custo RPA do ano)
        roi_ano = (economia_ano / custoComRPA_ano * 100) if custoComRPA_ano > 0 else 0.0
        roi_anos_percentuais.append(roi_ano)

        resultados_anuais.append({
            "Ano": ano,
            "Processos automatizados": processosAtivos,
            "Uso colab. por processo (%)": f"{usoAjustadoPercent:.1f}%",
            "FTE economizado (ano)": round(fteEconomizadosAno, 2),
            "Custo Sem RPA (ano)": formatarMoeda(custoSemRPA_ano),
            "Custo Com RPA (ano)": formatarMoeda(custoComRPA_ano),
            "Economia (ano)": formatarMoeda(economia_ano)
        })

    # =========================
    # Payback mensal REALISTA (rampa mensal + lead time curto)
    # =========================
    leadTimeMeses = 1  # ajuste conforme sua realidade (1–2 meses costuma fazer sentido)

    # Deriva novos processos por ano a partir da distribuição cumulativa
    novos_por_ano = []
    prev = 0
    for i in range(anos):
        curr = distribuicao[i]
        novos_por_ano.append(curr - prev)
        prev = curr

    # Distribuição mensal linear dos novos processos de cada ano
    ativacoes_por_mes = []
    for i in range(anos):
        novos = novos_por_ano[i]
        ativ_por_mes_ano = [novos / 12.0] * 12
        ativacoes_por_mes.extend(ativ_por_mes_ano)

    # Cumulativo de ativações sem lead
    cumulativo_ativ = []
    acum = 0.0
    for m in range(meses_totais):
        acum += ativacoes_por_mes[m]
        cumulativo_ativ.append(acum)

    # Processos ativos por mês aplicando lead time (shift) e respeitando o teto do ano
    ativos_por_mes = []
    for m in range(meses_totais):
        idx = m - leadTimeMeses
        ativos = cumulativo_ativ[idx] if idx >= 0 else 0.0
        ano_idx = m // 12
        max_ano = distribuicao[ano_idx]
        ativos = min(ativos, max_ano)
        ativos_por_mes.append(ativos)

    # Fatores mensais (geométricos)
    fatorVolumePorMes = fatorVolumePorAno ** (1 / 12) if anos > 1 else 1.0
    fatorCasosPorMes = fatorCasosPorAno ** (1 / 12) if anos > 1 else 1.0

    # Evolução mensal de demanda e casos (partindo dos valores iniciais)
    demandaMensal_mes = demandaMensalInicial
    casosPorExecucao_mes = casosPorExecucaoInicial

    economia_acumulada_mensal = 0.0
    for mes_idx in range(meses_totais):
        # Avança carga (mensal)
        if mes_idx > 0:
            demandaMensal_mes *= fatorVolumePorMes
            casosPorExecucao_mes *= fatorCasosPorMes

        # Carga relativa vs Ano 1 base
        carga_mes = demandaMensal_mes * casosPorExecucao_mes
        cargaRelativa_mes = (carga_mes / cargaBase) if cargaBase > 0 else 1.0

        # Uso por processo no mês (sem cap, mantendo sua filosofia)
        usoAjustadoFrac_mes = usoBaseFrac * cargaRelativa_mes

        # Custo manual do mês (proporcional aos ativos desse mês)
        processosAtivos_mes = ativos_por_mes[mes_idx]
        custoSemRPA_mes = processosAtivos_mes * custoColaborador * usoAjustadoFrac_mes * colaboradoresIniciais

        # Custo RPA do mês (sem diluição — custo da célula cheio)
        custoComRPA_mes = custoCelulaRPA_Mensal

        economia_mes = custoSemRPA_mes - custoComRPA_mes
        economia_acumulada_mensal += economia_mes

        if paybackMes is None and economia_acumulada_mensal >= 0:
            paybackMes = mes_idx + 1  # 1-based
            break

    # ROI e Payback (carteira)
    roiPercentual = (economiaTotal / custoTotalRPA * 100) if custoTotalRPA > 0 else 0.0
    paybackAnoStr = f"{paybackAno}º ano" if paybackAno is not None else "Não viável"
    paybackMesStr = f"{paybackMes}º mês" if paybackMes is not None else "Não viável"

    # Formatação pt-BR para ROI por ano (retirar zeros desnecessários e usar vírgula)
    def formatar_percentual_br(valor):
        s = f"{valor:.2f}".replace(".", ",")
        if s.endswith(",00"):
            s = s[:-3]
        return f"{s}%"

    roi_anos_formatados = [formatar_percentual_br(v) for v in roi_anos_percentuais]

    # Resumo financeiro e de horas
    resumoFinanceiro = {
        "Processos total (ao fim)": numProcessosAutomatizados,
        "Distribuição anual (processos ativos)": distribuicao,
        "Custo total manual (carteira ativa)": formatarMoeda(custoTotalManual),
        "Custo total RPA (carteira)": formatarMoeda(custoTotalRPA),
        "Economia total (carteira ativa)": formatarMoeda(economiaTotal),
        "FTE total (carteira ativa)": round(fteEconomizadosTotal, 2),
        "Payback (anos)": paybackAnoStr,
        "Payback (meses)": paybackMesStr,
        "ROI (carteira) (%)": formatar_percentual_br(roiPercentual),
        "ROI (anos) (%)": roi_anos_formatados
    }

    resumoHoras = (
        f"Horas totais manual (carteira ativa): {horasManualTotal:.1f}h\n"
        f"Horas totais RPA (carteira): {horasRPATotal:.1f}h\n"
        f"Horas totais economizadas (carteira): {(horasManualTotal - horasRPATotal):.1f}h"
    )

    return resultados_anuais, resumoFinanceiro, resumoHoras

# =========================
# OUTPUT
# =========================
dados, resumoFinanceiro, resumoHoras = analisarCarteiraAnualComRampa_UsoPercent(
    anos=anos, diasPorMes=diasPorMes, jornadaPadrao=jornadaPadrao,
    numProcessosAutomatizados=numProcessosAutomatizados, distribuicaoPorAno=distribuicaoPorAno,
    custoColaborador=custoColaborador, demandaMensalInicial=demandaMensalInicial,
    casosPorExecucaoInicial=casosPorExecucaoInicial, colaboradoresIniciais=colaboradoresIniciais,
    horasManualPorDia=horasManualPorDia, horasRPAPorDia=horasRPAPorDia,
    crescimentoVolume=crescimentoVolume, crescimentoCasos=crescimentoCasos,
    custoCelulaRPA_Mensal=custoCelulaRPA_Mensal
)

print(f"{'Ano':<6}{'Proc. automatizados':<22}{'Uso colab. p/ proc. (%)':<26}"
      f"{'FTE (ano)':<12}{'Custo Sem RPA (ano)':<24}{'Custo com RPA (ano)':<24}{'Economia (ano)':<18}")

for r in dados:
    print(f"{r['Ano']:<6}{r['Processos automatizados']:<22}"
          f"{r['Uso colab. por processo (%)']:<26}{r['FTE economizado (ano)']:<12}"
          f"{r['Custo Sem RPA (ano)']:<24}{r['Custo Com RPA (ano)']:<24}{r['Economia (ano)']:<18}")

print("\nResumo Financeiro da Carteira:")
for k, v in resumoFinanceiro.items():
    print(f"{k}: {v}")

print("\nResumo de Horas (Carteira):")
print(resumoHoras)