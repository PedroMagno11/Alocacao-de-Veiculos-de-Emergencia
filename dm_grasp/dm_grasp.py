from collections import Counter
import random
import time
import numpy
from config.PARAMETROS import (
    PARAMETRO_ALPHA, QUANTIDADE_MAXIMA_POR_TIPO,
    MAX_ITERACOES, MAX_ITERACOES_SEM_MELHORA,
    TIPOS_AMBULANCIA, SEMENTE_ALEATORIA,
)
from comum.avaliador_viabilidade import verificar_viabilidade
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
):
    melhor_solucao = set()
    melhor_fo = -1e18

    historico_fo = []
    historico_fo_antes_bl = []
    historico_fo_apos_bl = []

    print("=" * 70)
    print("FASE 1 — GRASP puro")
    print("=" * 70)

    for numero_iteracao in range(1, iteracoes_fase1 + 1):

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
            melhor_fo = fo_bl
            melhor_solucao = set(solucao_bl)

            cobertura = (
                100.0
                * len(obter_pontos_cobertos_pela_solucao(melhor_solucao, pontos_cobertos))
                / len(dataframe)
            )
            print(
                f"  [F1 iter {numero_iteracao:3d}] "
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
#  FASE 2 — DM-GRASP guiado com re-mineração
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
    max_tentativas_remineracao=5,
):
    melhor_fo = melhor_fo_fase1
    melhor_solucao = set(melhor_solucao_fase1)

    historico_fo = []
    historico_fo_antes_bl = []
    historico_fo_apos_bl = []

    # Mineração inicial com toda a elite da Fase 1
    pesos_candidatos = minerar_frequencia(
        memoria_elite,
        frequencia_minima=frequencia_minima,
    )

    print()
    print("=" * 70)
    print("FASE 2 — DM-GRASP guiado (re-mineração adaptativa)")
    print(f"  Pesos iniciais minerados: {len(pesos_candidatos)}")
    print("=" * 70)

    for numero_iteracao in range(1, iteracoes_fase2 + 1):

        # ── tentativas de construção com re-mineração ──────────────────────
        melhorou = False

        for tentativa in range(1, max_tentativas_remineracao + 1):

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

            solucao_bl, fo_bl = busca_local(
                solucao_construida,
                pontos_cobertos,
                regioes,
                tipos_ambulancia,
                quantidade_maxima_por_tipo,
                max_iteracoes_sem_melhora,
            )

            memoria_elite.adicionar(solucao_bl, fo_bl)

            if fo_bl > melhor_fo:
                melhor_fo = fo_bl
                melhor_solucao = set(solucao_bl)
                melhorou = True

                cobertura = (
                    100.0
                    * len(obter_pontos_cobertos_pela_solucao(melhor_solucao, pontos_cobertos))
                    / len(dataframe)
                )
                print(
                    f"  [F2 iter {numero_iteracao:3d} | tent {tentativa}] "
                    f"FO = {melhor_fo:.2f} | "
                    f"Alocações = {len(melhor_solucao)} | "
                    f"Cobertura = {cobertura:.1f}% | "
                    f"Pesos = {len(pesos_candidatos)}"
                )
                break   # melhora encontrada → segue para próxima iteração

            # Não melhorou → re-minera com a elite atualizada
            if tentativa < max_tentativas_remineracao:
                pesos_candidatos = minerar_frequencia(
                    memoria_elite,
                    frequencia_minima=frequencia_minima,
                )

        # Registra a melhor FO desta iteração (independente do nº de tentativas)
        historico_fo_antes_bl.append(fo_antes)
        historico_fo_apos_bl.append(fo_bl)
        historico_fo.append(fo_bl)

    return (
        melhor_solucao,
        melhor_fo,
        historico_fo,
        historico_fo_antes_bl,
        historico_fo_apos_bl,
    )


# ─────────────────────────────────────────────
#  ORQUESTRADOR PRINCIPAL
# ─────────────────────────────────────────────

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
    max_tentativas_remineracao=5,
):
    gerador_aleatorio = random.Random(semente_aleatoria)
    regioes = dataframe["local_id"].tolist()

    pontos_cobertos = pre_computar_pontos_cobertos(dataframe, tipos_ambulancia)
    memoria_elite = MemoriaElite(tamanho_maximo=tamanho_memoria_elite)

    tempo_inicio = time.time()

    # ── Fase 1 ────────────────────────────────────────────────────────────
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
    )

    # ── Fase 2 ────────────────────────────────────────────────────────────
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
        max_tentativas_remineracao=max_tentativas_remineracao,
    )

    tempo_total = time.time() - tempo_inicio

    # ── melhor global ─────────────────────────────────────────────────────
    if melhor_fo_f2 >= melhor_fo_f1:
        melhor_solucao = melhor_solucao_f2
        melhor_fo = melhor_fo_f2
    else:
        melhor_solucao = melhor_solucao_f1
        melhor_fo = melhor_fo_f1

    assert verificar_viabilidade(melhor_solucao, quantidade_maxima_por_tipo), (
        "ERRO: solução final inviável. "
        f"Contagem por tipo: {contar_ambulancias_por_tipo(melhor_solucao)}"
    )

    # ── estatísticas ──────────────────────────────────────────────────────
    historico_completo = historico_fo_f1 + historico_fo_f2
    arr = numpy.array(historico_completo)

    print()
    print("=" * 70)
    print("RESULTADO FINAL — DM-GRASP (2 fases)")
    print("=" * 70)
    print(f"Melhor FO Fase 1 : {melhor_fo_f1:.2f}")
    print(f"Melhor FO Fase 2 : {melhor_fo_f2:.2f}")
    print(f"Melhor FO global : {melhor_fo:.2f}")
    print(f"Total alocações  : {len(melhor_solucao)}")
    print(f"Tempo de execução: {tempo_total:.2f}s")
    print(f"Média FO (global): {arr.mean():.2f}")
    print(f"Desvio padrão    : {arr.std():.2f}")
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


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":

    dataframe = ler_instancia("instances/instancia.csv")

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
        tamanho_memoria_elite=20,
        frequencia_minima=0.3,
        max_tentativas_remineracao=5,
    )

    # # Plots com as duas fases distintas
    # plotar_funcao_objetivo_por_iteracao(
    #     historico_fo_f1,
    #     historico_fo_f2,
    # )

    plotar_comparacao_funcao_objetivo_antes_e_apos_busca_local(
        historico_antes_bl_f1 + historico_antes_bl_f2,
        historico_apos_bl_f1 + historico_apos_bl_f2,
    )