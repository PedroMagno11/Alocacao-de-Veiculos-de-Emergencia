"""
GRASP para o Problema de Máxima Cobertura com Sobreposição
para Frota de Ambulâncias (PMCS-FA)
"""
import time
import numpy
import random
 
from comum.avaliador_viabilidade import verificar_viabilidade
from config.PARAMETROS import (
    MAX_ITERACOES,
    MAX_ITERACOES_SEM_MELHORA,
    PARAMETRO_ALPHA,
    QUANTIDADE_MAXIMA_POR_TIPO,
    SEMENTE_ALEATORIA,
    TIPOS_AMBULANCIA,
)
from comum.busca_local import (
    busca_local,
    contar_ambulancias_por_tipo,
    obter_pontos_cobertos_pela_solucao,
    calcular_funcao_objetivo,
)
from grasp.construtivo_grasp import construir_solucao
from plots.visualizacao import (
    plotar_funcao_objetivo_por_iteracao,
    plotar_comparacao_funcao_objetivo_antes_e_apos_busca_local,
)
from instances.read_instance import ler_instancia, pre_computar_pontos_cobertos
 
# ============================================================
#  GRASP
# ============================================================
def executar_grasp(dataframe, tipos_ambulancia, quantidade_maxima_por_tipo,
                   parametro_alpha, max_iteracoes, max_iteracoes_sem_melhora,
                   semente_aleatoria, tempo_limite_s = None):
    
    gerador_aleatorio = random.Random(semente_aleatoria)
    regioes           = dataframe["local_id"].tolist()

    # ── Cabecalho ─────────────────────────────────────────────
    separador = "=" * 70
    print(separador)
    print("  GRASP — Problema de Maxima Cobertura com Sobreposicao (PMCS-FA)")
    print(separador)
    print(f"  Pontos de demanda        : {len(dataframe)}")
    print(f"  Regioes base candidatas  : {len(dataframe)}")
    for tipo, configuracao in tipos_ambulancia.items():
        print(
            f"  Tipo {tipo} ({configuracao['nome']:8s}): "
            f"raio = {configuracao['raio_cobertura_km']} km, "
            f"quantidade maxima = {quantidade_maxima_por_tipo[tipo]}"
        )
    print(f"  Parametro alpha (RCL)    : {parametro_alpha}  "
          f"(0 = guloso puro, 1 = aleatorio puro)")
    print(f"  Iteracoes maximas        : {max_iteracoes}")

    if tempo_limite_s is not None:
        print(f"  Tempo limite             : {tempo_limite_s}s")
    print(f"  Parada busca local       : {max_iteracoes_sem_melhora} iteracoes sem melhora")
    print()
 
    # ── Pré-processamento ──────────────────────────────────────
    print("  Carregando pontos cobertos do dataframe ...", end=" ", flush=True)
    t_pre = time.time()
    pontos_cobertos = pre_computar_pontos_cobertos(dataframe, tipos_ambulancia)
    print(f"concluido em {time.time() - t_pre:.1f}s\n")

    melhor_solucao   = set()
    melhor_fo        = -1e18
    historico_fo     = []
    historico_fo_antes_busca_local = []
    historico_fo_pos_busca_local = []

    tempo_inicio_grasp = time.time()

    for numero_iteracao in range(1, max_iteracoes + 1):

        if tempo_limite_s is not None:
            tempo_decorrido = time.time() - tempo_inicio_grasp
            if tempo_decorrido >= tempo_limite_s:
                print(
                    f"\n  [GRASP] Tempo limite atingido após "
                    f"{numero_iteracao - 1} iterações "
                    f"({tempo_decorrido:.1f}s >= {tempo_limite_s}s)."
                )
                break

        # construtivo GRASP
        solucao_construida = construir_solucao(
            pontos_cobertos, regioes, tipos_ambulancia,
            quantidade_maxima_por_tipo, parametro_alpha, gerador_aleatorio
        )

        fo_antes_busca_local = calcular_funcao_objetivo(pontos_cobertos=pontos_cobertos, solucao=solucao_construida)
        
        historico_fo_antes_busca_local.append(fo_antes_busca_local)

        # busca local
        solucao_apos_busca_local, fo_apos_busca_local = busca_local(
            solucao_construida, pontos_cobertos, regioes,
            tipos_ambulancia, quantidade_maxima_por_tipo,
            max_iteracoes_sem_melhora
        )

        historico_fo_pos_busca_local.append(fo_apos_busca_local)

        historico_fo.append(fo_antes_busca_local)
        historico_fo.append(fo_apos_busca_local)

        # Atualiza melhor solucao global
        if fo_apos_busca_local > melhor_fo:
            melhor_fo      = fo_apos_busca_local
            melhor_solucao = set(solucao_apos_busca_local)
            percentual_cobertura = (
                100.0 * len(obter_pontos_cobertos_pela_solucao(melhor_solucao, pontos_cobertos))
                / len(dataframe)
            )

            tempo_ate_melhora = time.time() - tempo_inicio_grasp

            print(
                f"  [iter {numero_iteracao:4d} | {tempo_ate_melhora:6.1f}s]  "
                f"FO = {melhor_fo:8.2f}  "
                f"Alocações = {len(melhor_solucao):2d}  "
                f"Cobertura = {percentual_cobertura:.1f}%"
            )

    tempo_total = time.time() - tempo_inicio_grasp

    # ── Verificacao de viabilidade ────────────────────────────
    assert verificar_viabilidade(melhor_solucao, quantidade_maxima_por_tipo), (
        f"ERRO: solucao final e inviavel! "
        f"Contagem por tipo: {contar_ambulancias_por_tipo(melhor_solucao)}"
    )

    pontos_cobertos_solucao = obter_pontos_cobertos_pela_solucao(
        melhor_solucao, pontos_cobertos
    )
    contagem_final = contar_ambulancias_por_tipo(melhor_solucao)
    array_historico = numpy.array(historico_fo)

    # ── Resultado final ───────────────────────────────────────
    print()
    print(separador)
    print("  RESULTADO FINAL - GRASP")
    print(separador)
    print(f"  Iterações executadas     : {len(historico_fo)}")
    print(f"  Tempo de execução        : {tempo_total:.1f}s")
    print(f"  Funcao objetivo          : {melhor_fo:.2f}")
    print(
        f"  Pontos cobertos          : "
        f"{len(pontos_cobertos_solucao)} de {len(dataframe)}  "
        f"({100.0 * len(pontos_cobertos_solucao) / len(dataframe):.1f}%)"
    )
    print(f"  Total de alocacoes       : {len(melhor_solucao)}")
    print()
    print("  Estatísticas das iterações:")
    print(f"    Melhor FO : {array_historico.max():.2f}")
    print(f"    Média FO  : {array_historico.mean():.2f}")
    print(f"    Pior  FO  : {array_historico.min():.2f}")
    print(f"    Desvio dp : {array_historico.std():.2f}")
    print(separador)

    return (
        melhor_solucao, 
        melhor_fo, 
        pontos_cobertos, 
        historico_fo, 
        historico_fo_antes_busca_local, 
        historico_fo_pos_busca_local,
    )


if __name__ == "__main__":
    dataframe = ler_instancia("instances/instancia_aleatoria_01_100p.csv")

    # dataframe = ler_instancia("instances/instancia.csv")

    melhor_solucao, melhor_fo, pontos_cobertos, historico_fo, historico_pre_busca_local, historico_pos_busca_local = executar_grasp(
        dataframe                  = dataframe,
        tipos_ambulancia           = TIPOS_AMBULANCIA,
        quantidade_maxima_por_tipo = QUANTIDADE_MAXIMA_POR_TIPO,
        parametro_alpha            = PARAMETRO_ALPHA,
        max_iteracoes              = MAX_ITERACOES,
        max_iteracoes_sem_melhora  = MAX_ITERACOES_SEM_MELHORA,
        semente_aleatoria          = SEMENTE_ALEATORIA,
        tempo_limite_s= 3600, # 1 hora em segundos
    )

    plotar_funcao_objetivo_por_iteracao(
        historico_fo
    )

    plotar_comparacao_funcao_objetivo_antes_e_apos_busca_local(
        historico_fo_antes_busca_local_f1=historico_pre_busca_local, historico_fo_pos_busca_local_f1=historico_pos_busca_local
    )