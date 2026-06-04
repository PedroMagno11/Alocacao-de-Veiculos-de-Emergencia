from time import perf_counter
import random

import config.PARAMETROS as PARAMETROS
from comum.avaliador_viabilidade import verificar_viabilidade
from comum.busca_local import (
    busca_local,
    calcular_funcao_objetivo,
    obter_pontos_cobertos_pela_solucao,
    pre_computar_bitsets,
)
from plots.visualizacao import plotar_solucoes_background, plotar_cobertura, plotar_solucoes, plotar_movimentos_busca_local, plotar_evolucao_grasp
from grasp.construtivo_grasp import construir_solucao
from instances.read_instance import ler_instancia, pre_computar_pontos_cobertos

# CAMINHO_INSTANCIA = "instancia.csv"

CAMINHO_INSTANCIA = "instances/instancia_aleatoria_01_500p.csv"


def main():
    tempo_inicio_total = perf_counter()

    dataframe = ler_instancia(CAMINHO_INSTANCIA)

    print("Instancia carregada com sucesso.")
    print(f"Arquivo: {CAMINHO_INSTANCIA}")
    print(f"Quantidade de pontos/regioes: {len(dataframe)}")
    print(f"Colunas: {', '.join(dataframe.columns)}")

    tempo_inicio = perf_counter()
    pontos_cobertos = pre_computar_pontos_cobertos(
        dataframe,
        PARAMETROS.TIPOS_AMBULANCIA,
    )
    tempo_leitura_cobertura = perf_counter() - tempo_inicio

    regioes = dataframe["local_id"].to_list()
    gerador_aleatorio = random.Random(PARAMETROS.SEMENTE_ALEATORIA)

    tempo_inicio = perf_counter()
    pre_calculo_cobertura = pre_computar_bitsets(
        pontos_cobertos=pontos_cobertos,
        regioes=regioes,
        tipos_ambulancia=PARAMETROS.TIPOS_AMBULANCIA,
        # Use True apenas se quiser materializar todas as intersecoes em memoria.
        # Para instancias grandes, False costuma ser mais seguro e ainda rapido.
        pre_computar_todas_intersecoes=False,
    )
    tempo_pre_calculo_bitset = perf_counter() - tempo_inicio

    tempo_inicio = perf_counter()
    solucao = construir_solucao(
        pontos_cobertos,
        regioes,
        PARAMETROS.TIPOS_AMBULANCIA,
        PARAMETROS.QUANTIDADE_MAXIMA_POR_TIPO,
        PARAMETROS.PARAMETRO_ALPHA,
        gerador_aleatorio,
    )
    tempo_construtivo = perf_counter() - tempo_inicio

    fo_inicial = calcular_funcao_objetivo(solucao, pre_calculo=pre_calculo_cobertura)

    print("\n--- Solucao inicial ---")
    print(f"Viavel: {verificar_viabilidade(solucao, PARAMETROS.QUANTIDADE_MAXIMA_POR_TIPO)}")
    print(f"Solucao: {solucao}")
    print(f"Funcao objetivo inicial: {fo_inicial}")
    print(
        "Pontos fisicamente cobertos: "
        f"{len(obter_pontos_cobertos_pela_solucao(solucao, pre_calculo=pre_calculo_cobertura))}"
    )

    tempo_inicio = perf_counter()
    solucao_com_busca_local, fo_busca_local = busca_local(
        solucao_inicial=solucao,
        pontos_cobertos=pontos_cobertos,
        regioes=regioes,
        tipos_ambulancia=PARAMETROS.TIPOS_AMBULANCIA,
        quantidade_maxima_por_tipo=PARAMETROS.QUANTIDADE_MAXIMA_POR_TIPO,
        max_iteracoes_sem_melhora=PARAMETROS.MAX_ITERACOES_SEM_MELHORA,
        pre_calculo=pre_calculo_cobertura,
    )
    tempo_busca_local = perf_counter() - tempo_inicio

    print("\n--- Solucao apos busca local ---")
    print(f"Solucao: {solucao_com_busca_local}")
    print(f"Funcao objetivo busca local: {fo_busca_local}")
    print(
        "Pontos fisicamente cobertos: "
        f"{len(obter_pontos_cobertos_pela_solucao(solucao_com_busca_local, pre_calculo=pre_calculo_cobertura))}"
    )

    print("\n--- Tempos de execucao ---")
    print(f"Leitura das coberturas: {tempo_leitura_cobertura:.6f} s")
    print(f"Pre-calculo bitset:     {tempo_pre_calculo_bitset:.6f} s")
    print(f"Construtivo:            {tempo_construtivo:.6f} s")
    print(f"Busca local:            {tempo_busca_local:.6f} s")
    print(f"Total:                  {perf_counter() - tempo_inicio_total:.6f} s")
    print(f"Intersecoes cacheadas:  {len(pre_calculo_cobertura.cache_intersecao)}")


    plotar_solucoes_background(dataframe=dataframe, solucao_construtivo=solucao, solucao_busca_local=solucao_com_busca_local)

    plotar_solucoes(
        dataframe,
        solucao,
        solucao_com_busca_local
    )

    plotar_movimentos_busca_local(
        dataframe,
        solucao,
        solucao_com_busca_local
    )

    # plotar_evolucao_grasp(
    #     historico_fo
    # )

if __name__ == "__main__":
    main()
