"""
DM-GRASP para o Problema de Máxima Cobertura com Sobreposição
para Frota de Ambulâncias (PMCS-FA)
"""

import time
import numpy
import random

from comum.avaliador_viabilidade import verificar_viabilidade
from config.PARAMETROS import (
    PARAMETRO_ALPHA,
    QUANTIDADE_MAXIMA_POR_TIPO,
    MAX_ITERACOES,
    MAX_ITERACOES_SEM_MELHORA,
    TIPOS_AMBULANCIA,
    SEMENTE_ALEATORIA,
)
from comum.busca_local import (
    busca_local,
    contar_ambulancias_por_tipo,
    obter_pontos_cobertos_pela_solucao,
    calcular_funcao_objetivo,
)
from dm_grasp.construtivo_dm_grasp import construir_solucao_dm
from dm_grasp.memoria_elite import MemoriaElite
from dm_grasp.minerador_frequencia import minerar_frequencia
from instances.read_instance import pre_computar_pontos_cobertos, ler_instancia
from plots.visualizacao import (
    plotar_comparacao_funcao_objetivo_antes_e_apos_busca_local,
    plotar_funcao_objetivo_por_iteracao,
)


# ─────────────────────────────────────────────
#  FASE 1 — GRASP puro (sem mineração)
# ─────────────────────────────────────────────

def executar_fase1(
    pontos_cobertos,
    regioes,
    tipos_ambulancia,
    quantidade_maxima_por_tipo,
    parametro_alpha,
    max_iteracoes_sem_melhora,
    iteracoes_fase1,
    gerador_aleatorio,
    memoria_elite,
    dataframe,
    tempo_inicio_global,          # <-- referência de tempo do início total
    tempo_limite_fase1_s=None,    # <-- None = sem limite de tempo
):
    melhor_solucao = set()
    melhor_fo      = -1e18

    historico_fo          = []
    historico_fo_antes_bl = []
    historico_fo_apos_bl  = []

    print("=" * 70)
    print("FASE 1 — GRASP puro")
    if tempo_limite_fase1_s is not None:
        print(f"  Orçamento de tempo: {tempo_limite_fase1_s:.1f}s")
    print("=" * 70)

    for numero_iteracao in range(1, iteracoes_fase1 + 1):

        # ── Critério de parada por tempo ──────────────────────
        if tempo_limite_fase1_s is not None:
            tempo_fase1 = time.time() - tempo_inicio_global
            if tempo_fase1 >= tempo_limite_fase1_s:
                print(
                    f"  [F1] Tempo limite da fase atingido após "
                    f"{numero_iteracao - 1} iterações "
                    f"({tempo_fase1:.1f}s >= {tempo_limite_fase1_s:.1f}s)."
                )
                break

        solucao_construida = construir_solucao_dm(
            pontos_cobertos=pontos_cobertos,
            regioes=regioes,
            tipos_ambulancia=tipos_ambulancia,
            quantidade_maxima_por_tipo=quantidade_maxima_por_tipo,
            parametro_alpha=parametro_alpha,
            gerador_aleatorio=gerador_aleatorio,
            pesos_candidatos={},          # sem viés na Fase 1
        )

        fo_antes = calcular_funcao_objetivo(
            solucao=solucao_construida,
            pontos_cobertos=pontos_cobertos,
        )
        historico_fo_antes_bl.append(fo_antes)

        solucao_bl, fo_bl = busca_local(
            solucao_construida,
            pontos_cobertos,
            regioes,
            tipos_ambulancia,
            quantidade_maxima_por_tipo,
            max_iteracoes_sem_melhora,
        )

        historico_fo_apos_bl.append(fo_bl)
        historico_fo.append(fo_bl)
        memoria_elite.adicionar(solucao_bl, fo_bl)

        if fo_bl > melhor_fo:
            melhor_fo      = fo_bl
            melhor_solucao = set(solucao_bl)

            cobertura = (
                100.0
                * len(obter_pontos_cobertos_pela_solucao(melhor_solucao, pontos_cobertos))
                / len(dataframe)
            )
            tempo_decorrido = time.time() - tempo_inicio_global
            print(
                f"  [F1 iter {numero_iteracao:3d} | {tempo_decorrido:6.1f}s] "
                f"FO = {melhor_fo:.2f} | "
                f"Alocações = {len(melhor_solucao)} | "
                f"Cobertura = {cobertura:.1f}%"
            )

    return (
        melhor_solucao,
        melhor_fo,
        historico_fo,
        historico_fo_antes_bl,
        historico_fo_apos_bl,
    )


# ─────────────────────────────────────────────
#  FASE 2 — DM-GRASP guiado (mineração única)
# ─────────────────────────────────────────────

def executar_fase2(
    pontos_cobertos,
    regioes,
    tipos_ambulancia,
    quantidade_maxima_por_tipo,
    parametro_alpha,
    max_iteracoes_sem_melhora,
    iteracoes_fase2,
    gerador_aleatorio,
    memoria_elite,
    melhor_fo_fase1,
    melhor_solucao_fase1,
    dataframe,
    frequencia_minima=0.2,
    tempo_inicio_global=None,     # <-- referência de tempo do início total
    tempo_limite_total_s=None,    # <-- orçamento total (fase 1 + fase 2)
):
    melhor_fo      = melhor_fo_fase1
    melhor_solucao = set(melhor_solucao_fase1)

    historico_fo          = []
    historico_fo_antes_bl = []
    historico_fo_apos_bl  = []

    # Mineração feita uma só vez com a elite da Fase 1
    pesos_candidatos = minerar_frequencia(
        memoria_elite,
        frequencia_minima=frequencia_minima,
    )

    print()
    print("=" * 70)
    print("FASE 2 — DM-GRASP guiado (mineração única)")
    quantidade_padroes = len(pesos_candidatos)
    quantidade_itens = getattr(
        pesos_candidatos,
        "quantidade_itens_ponderados",
        len(pesos_candidatos),
    )
    suporte_minimo = getattr(pesos_candidatos, "suporte_minimo", None)

    print(f"  Padroes maximos minerados: {quantidade_padroes}")
    print(f"  Itens ponderados: {quantidade_itens}")
    if suporte_minimo is not None:
        print(f"  Suporte minimo absoluto: {suporte_minimo}")
    if tempo_limite_total_s is not None and tempo_inicio_global is not None:
        tempo_restante = tempo_limite_total_s - (time.time() - tempo_inicio_global)
        print(f"  Tempo restante para Fase 2: {max(tempo_restante, 0):.1f}s")
    print("=" * 70)

    for numero_iteracao in range(1, iteracoes_fase2 + 1):

        # ── Critério de parada por tempo ──────────────────────
        if tempo_limite_total_s is not None and tempo_inicio_global is not None:
            tempo_decorrido = time.time() - tempo_inicio_global
            if tempo_decorrido >= tempo_limite_total_s:
                print(
                    f"\n  [F2] Tempo limite total atingido após "
                    f"{numero_iteracao - 1} iterações da Fase 2 "
                    f"({tempo_decorrido:.1f}s >= {tempo_limite_total_s:.1f}s)."
                )
                break

        solucao_construida = construir_solucao_dm(
            pontos_cobertos=pontos_cobertos,
            regioes=regioes,
            tipos_ambulancia=tipos_ambulancia,
            quantidade_maxima_por_tipo=quantidade_maxima_por_tipo,
            parametro_alpha=parametro_alpha,
            gerador_aleatorio=gerador_aleatorio,
            pesos_candidatos=pesos_candidatos,
        )

        fo_antes = calcular_funcao_objetivo(
            solucao=solucao_construida,
            pontos_cobertos=pontos_cobertos,
        )
        historico_fo_antes_bl.append(fo_antes)

        solucao_bl, fo_bl = busca_local(
            solucao_construida,
            pontos_cobertos,
            regioes,
            tipos_ambulancia,
            quantidade_maxima_por_tipo,
            max_iteracoes_sem_melhora,
        )

        historico_fo_apos_bl.append(fo_bl)
        historico_fo.append(fo_bl)
        memoria_elite.adicionar(solucao_bl, fo_bl)

        if fo_bl > melhor_fo:
            melhor_fo      = fo_bl
            melhor_solucao = set(solucao_bl)

            cobertura = (
                100.0
                * len(obter_pontos_cobertos_pela_solucao(melhor_solucao, pontos_cobertos))
                / len(dataframe)
            )
            tempo_decorrido = time.time() - tempo_inicio_global if tempo_inicio_global else 0
            print(
                f"  [F2 iter {numero_iteracao:3d} | {tempo_decorrido:6.1f}s] "
                f"FO = {melhor_fo:.2f} | "
                f"Alocações = {len(melhor_solucao)} | "
                f"Cobertura = {cobertura:.1f}%"
            )

    return (
        melhor_solucao,
        melhor_fo,
        historico_fo,
        historico_fo_antes_bl,
        historico_fo_apos_bl,
    )


def executar_dm_grasp(
    dataframe,
    tipos_ambulancia,
    quantidade_maxima_por_tipo,
    parametro_alpha,
    iteracoes_fase1,
    iteracoes_fase2,
    max_iteracoes_sem_melhora,
    semente_aleatoria,
    tamanho_memoria_elite=50,
    frequencia_minima=0.2,
    tempo_limite_s=None,           # <-- None = sem limite de tempo
    proporcao_tempo_fase1=0.4,     # <-- 40% do tempo para Fase 1
):
    """
    Executa o DM-GRASP para o PMCS-FA.

    Critério de parada:
      - Por iterações (padrão): Fase 1 roda `iteracoes_fase1` iterações;
        Fase 2 roda `iteracoes_fase2` iterações.
      - Por tempo (`tempo_limite_s` definido): o orçamento total é dividido
        entre as fases pela `proporcao_tempo_fase1` (padrão 40/60).
        O que esgotar primeiro (iterações ou tempo) encerra a fase.

    """

    gerador_aleatorio = random.Random(semente_aleatoria)
    regioes = dataframe["local_id"].tolist()

    pontos_cobertos = pre_computar_pontos_cobertos(dataframe, tipos_ambulancia)
    memoria_elite   = MemoriaElite(tamanho_maximo=tamanho_memoria_elite)

    # Calcula orçamentos de tempo por fase
    tempo_limite_fase1 = None
    if tempo_limite_s is not None:
        tempo_limite_fase1 = tempo_limite_s * proporcao_tempo_fase1
        print(
            f"  Orçamento total  : {tempo_limite_s}s  "
            f"(Fase 1: {tempo_limite_fase1:.1f}s | "
            f"Fase 2: {tempo_limite_s - tempo_limite_fase1:.1f}s)"
        )

    tempo_inicio = time.time()

    # ── Fase 1 ────────────────────────────────────────────────
    (
        melhor_solucao_f1,
        melhor_fo_f1,
        historico_fo_f1,
        historico_antes_bl_f1,
        historico_apos_bl_f1,
    ) = executar_fase1(
        pontos_cobertos=pontos_cobertos,
        regioes=regioes,
        tipos_ambulancia=tipos_ambulancia,
        quantidade_maxima_por_tipo=quantidade_maxima_por_tipo,
        parametro_alpha=parametro_alpha,
        max_iteracoes_sem_melhora=max_iteracoes_sem_melhora,
        iteracoes_fase1=iteracoes_fase1,
        gerador_aleatorio=gerador_aleatorio,
        memoria_elite=memoria_elite,
        dataframe=dataframe,
        tempo_inicio_global=tempo_inicio,
        tempo_limite_fase1_s=tempo_limite_fase1,
    )

    # ── Fase 2 ────────────────────────────────────────────────
    (
        melhor_solucao_f2,
        melhor_fo_f2,
        historico_fo_f2,
        historico_antes_bl_f2,
        historico_apos_bl_f2,
    ) = executar_fase2(
        pontos_cobertos=pontos_cobertos,
        regioes=regioes,
        tipos_ambulancia=tipos_ambulancia,
        quantidade_maxima_por_tipo=quantidade_maxima_por_tipo,
        parametro_alpha=parametro_alpha,
        max_iteracoes_sem_melhora=max_iteracoes_sem_melhora,
        iteracoes_fase2=iteracoes_fase2,
        gerador_aleatorio=gerador_aleatorio,
        memoria_elite=memoria_elite,
        melhor_fo_fase1=melhor_fo_f1,
        melhor_solucao_fase1=melhor_solucao_f1,
        dataframe=dataframe,
        frequencia_minima=frequencia_minima,
        tempo_inicio_global=tempo_inicio,
        tempo_limite_total_s=tempo_limite_s,       # tempo total — a fase 2 usa o restante
    )

    tempo_total = time.time() - tempo_inicio

    # ── Melhor global ──────────────────────────────────────────
    if melhor_fo_f2 >= melhor_fo_f1:
        melhor_solucao = melhor_solucao_f2
        melhor_fo      = melhor_fo_f2
    else:
        melhor_solucao = melhor_solucao_f1
        melhor_fo      = melhor_fo_f1

    assert verificar_viabilidade(melhor_solucao, quantidade_maxima_por_tipo), (
        "ERRO: solução final inviável. "
        f"Contagem por tipo: {contar_ambulancias_por_tipo(melhor_solucao)}"
    )

    # ── Estatísticas ───────────────────────────────────────────
    historico_completo = historico_fo_f1 + historico_fo_f2
    arr = numpy.array(historico_completo)

    print()
    print("=" * 70)
    print("RESULTADO FINAL — DM-GRASP (2 fases)")
    print("=" * 70)
    print(f"  Iterações Fase 1  : {len(historico_fo_f1)}")
    print(f"  Iterações Fase 2  : {len(historico_fo_f2)}")
    print(f"  Melhor FO Fase 1  : {melhor_fo_f1:.2f}")
    print(f"  Melhor FO Fase 2  : {melhor_fo_f2:.2f}")
    print(f"  Melhor FO global  : {melhor_fo:.2f}")
    print(f"  Total alocações   : {len(melhor_solucao)}")
    print(f"  Tempo de execução : {tempo_total:.2f}s")
    print(f"  Média FO (global) : {arr.mean():.2f}")
    print(f"  Desvio padrão     : {arr.std():.2f}")
    print("=" * 70)

    return (
        melhor_solucao,
        melhor_fo,
        pontos_cobertos,
        historico_fo_f1,
        historico_fo_f2,
        historico_antes_bl_f1,
        historico_apos_bl_f1,
        historico_antes_bl_f2,
        historico_apos_bl_f2,
    )


if __name__ == "__main__":

    # dataframe = ler_instancia("instances/instancia.csv")
    dataframe = ler_instancia("instances/instancia_aleatoria_01_1000p.csv")

    (
        melhor_solucao,
        melhor_fo,
        pontos_cobertos,
        historico_fo_f1,
        historico_fo_f2,
        historico_antes_bl_f1,
        historico_apos_bl_f1,
        historico_antes_bl_f2,
        historico_apos_bl_f2,
    ) = executar_dm_grasp(
        dataframe=dataframe,
        tipos_ambulancia=TIPOS_AMBULANCIA,
        quantidade_maxima_por_tipo=QUANTIDADE_MAXIMA_POR_TIPO,
        parametro_alpha=PARAMETRO_ALPHA,
        iteracoes_fase1=100,
        iteracoes_fase2=200,
        max_iteracoes_sem_melhora=MAX_ITERACOES_SEM_MELHORA,
        semente_aleatoria=SEMENTE_ALEATORIA,
        tamanho_memoria_elite=40,
        frequencia_minima=0.3,
        tempo_limite_s=3600,          # 1 hora no total
        proporcao_tempo_fase1=0.4,    # 40% para Fase 1, 60% para Fase 2
    )

    plotar_comparacao_funcao_objetivo_antes_e_apos_busca_local(
        historico_antes_bl_f1 + historico_antes_bl_f2,
        historico_apos_bl_f1  + historico_apos_bl_f2,
    )
