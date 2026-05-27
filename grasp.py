"""
GRASP para o Problema de Máxima Cobertura com Sobreposição
para Frota de Ambulâncias (PMCS-FA)
"""

import numpy
import random
import time

from avaliador_viabilidade import verificar_viabilidade

from PARAMETROS import (
    MAX_ITERACOES,
    MAX_ITERACOES_SEM_MELHORA,
    PARAMETRO_ALPHA,
    QUANTIDADE_MAXIMA_POR_TIPO,
    SEMENTE_ALEATORIA,
    TIPOS_AMBULANCIA,
)
from busca_local import (
    busca_local,
    contar_ambulancias_por_tipo,
    obter_pontos_cobertos_pela_solucao,
)
from construtivo import construir_solucao
from read_instance import (
    ler_instancia,
    pre_computar_pontos_cobertos,
)

# ============================================================
#  GRASP
# ============================================================
def executar_grasp(dataframe, tipos_ambulancia, quantidade_maxima_por_tipo,
                   parametro_alpha, max_iteracoes, max_iteracoes_sem_melhora,
                   semente_aleatoria):
    
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
    print(f"  Iteracoes GRASP          : {max_iteracoes}")
    print(f"  Parada busca local       : {max_iteracoes_sem_melhora} iteracoes sem melhora")
    print()

    # ── Pre-processamento ──────────────────────────────────────
    print("  Carregando pontos cobertos do dataframe ...", end=" ", flush=True)
    tempo_inicio_precomputo = time.time()
    pontos_cobertos = pre_computar_pontos_cobertos(dataframe, tipos_ambulancia)
    tempo_precomputo = time.time() - tempo_inicio_precomputo
    print(f"concluido em {tempo_precomputo:.1f}s\n")

    melhor_solucao   = set()
    melhor_fo        = -1e18
    historico_fo     = []
    tempo_inicio_grasp = time.time()

    for numero_iteracao in range(1, max_iteracoes + 1):

        # construtivo GRASP
        solucao_construida = construir_solucao(
            pontos_cobertos, regioes, tipos_ambulancia,
            quantidade_maxima_por_tipo, parametro_alpha, gerador_aleatorio
        )

        # busca local
        solucao_apos_busca_local, fo_apos_busca_local = busca_local(
            solucao_construida, pontos_cobertos, regioes,
            tipos_ambulancia, quantidade_maxima_por_tipo,
            max_iteracoes_sem_melhora
        )

        historico_fo.append(fo_apos_busca_local)

        # Atualiza melhor solucao global
        if fo_apos_busca_local > melhor_fo:
            melhor_fo      = fo_apos_busca_local
            melhor_solucao = set(solucao_apos_busca_local)
            percentual_cobertura = (
                100.0 * len(obter_pontos_cobertos_pela_solucao(melhor_solucao, pontos_cobertos))
                / len(dataframe)
            )
            print(
                f"  [Iteracao {numero_iteracao:3d}]  "
                f"Funcao objetivo = {melhor_fo:8.2f}  "
                f"Alocacoes = {len(melhor_solucao):2d}  "
                f"Cobertura = {percentual_cobertura:.1f}%"
            )
        melhor_solucao = solucao_construida
        print(
                f"  [Iteracao {numero_iteracao:3d}]  "
                # f"Funcao objetivo = {melhor_fo:8.2f}  "
                f"Alocacoes = {melhor_solucao}  "
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
    print("  RESULTADO FINAL")
    print(separador)
    print(f"  Funcao objetivo          : {melhor_fo:.2f}")
    print(
        f"  Pontos cobertos          : "
        f"{len(pontos_cobertos_solucao)} de {len(dataframe)}  "
        f"({100.0 * len(pontos_cobertos_solucao) / len(dataframe):.1f}%)"
    )
    print(f"  Total de alocacoes       : {len(melhor_solucao)}")
    print(f"  Tempo de execucao        : {tempo_total:.1f}s")
    print()

    # Tabela de alocacoes
    cabecalho_tabela = (
        f"  {'Regiao':>7}  {'Tipo':>4}  {'Nome':10}  "
        f"{'Latitude':>12}  {'Longitude':>13}  {'Pontos cobertos':>15}"
    )
    print(cabecalho_tabela)
    print("  " + "-" * (len(cabecalho_tabela) - 2))
    for id_regiao, tipo in sorted(melhor_solucao):
        linha = dataframe[dataframe["local_id"] == id_regiao].iloc[0]
        print(
            f"  {id_regiao:7d}  {tipo:4d}  "
            f"{tipos_ambulancia[tipo]['nome']:10}  "
            f"{linha['latitude']:12.5f}  {linha['longitude']:13.5f}  "
            f"{len(pontos_cobertos[id_regiao][tipo]):15d}"
        )

    print()
    print("  Ambulancias utilizadas por tipo:")
    for tipo, configuracao in tipos_ambulancia.items():
        utilizadas  = contagem_final.get(tipo, 0)
        disponivel  = quantidade_maxima_por_tipo[tipo]
        barra       = "█" * utilizadas + "░" * (disponivel - utilizadas)
        print(
            f"    Tipo {tipo} ({configuracao['nome']:8s}): "
            f"{utilizadas} de {disponivel}  [{barra}]"
        )

    print()
    print("  Estatisticas das iteracoes do GRASP:")
    print(f"    Melhor funcao objetivo   : {array_historico.max():.2f}")
    print(f"    Media da funcao objetivo : {array_historico.mean():.2f}")
    print(f"    Pior funcao objetivo     : {array_historico.min():.2f}")
    print(f"    Desvio padrao            : {array_historico.std():.2f}")
    print(separador)

    return melhor_solucao, melhor_fo, pontos_cobertos, historico_fo


if __name__ == "__main__":
    dataframe = ler_instancia("instancia.csv")

    melhor_solucao, melhor_fo, pontos_cobertos, historico_fo = executar_grasp(
        dataframe                  = dataframe,
        tipos_ambulancia           = TIPOS_AMBULANCIA,
        quantidade_maxima_por_tipo = QUANTIDADE_MAXIMA_POR_TIPO,
        parametro_alpha            = PARAMETRO_ALPHA,
        max_iteracoes              = MAX_ITERACOES,
        max_iteracoes_sem_melhora  = MAX_ITERACOES_SEM_MELHORA,
        semente_aleatoria          = SEMENTE_ALEATORIA,
    )
