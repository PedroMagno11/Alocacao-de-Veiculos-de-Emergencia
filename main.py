from read_instance import ler_instancia, pre_computar_pontos_cobertos
import PARAMETROS
from construtivo import construir_solucao
from avaliador_viabilidade import verificar_viabilidade
from busca_local import (
    obter_pontos_cobertos_pela_solucao,
    pre_computar_areas_e_intersecoes,
    calcular_funcao_objetivo,
    busca_local,
)
import random

CAMINHO_INSTANCIA = "instancia.csv"


def main():
    dataframe = ler_instancia(CAMINHO_INSTANCIA)

    print("Instancia carregada com sucesso.")
    print(f"Arquivo: {CAMINHO_INSTANCIA}")
    print(f"Quantidade de pontos/regioes: {len(dataframe)}")
    print(f"Colunas: {', '.join(dataframe.columns)}")
    
    
    pontos_cobertos = pre_computar_pontos_cobertos(
        dataframe,
        PARAMETROS.TIPOS_AMBULANCIA,
    )

    regioes = dataframe["local_id"].to_list()
    gerador_aleatorio = random.Random(PARAMETROS.SEMENTE_ALEATORIA)

    # Pre-calculo em memoria das areas cobertas e intersecoes.
    # Essa estrutura e reutilizada pela funcao objetivo e pela busca local,
    # evitando recalcular intersecoes durante a exploracao da vizinhanca.
    pre_calculo_cobertura = pre_computar_areas_e_intersecoes(
        pontos_cobertos=pontos_cobertos,
        regioes=regioes,
        tipos_ambulancia=PARAMETROS.TIPOS_AMBULANCIA,
    )

    solucao = construir_solucao(
        pontos_cobertos,
        regioes,
        PARAMETROS.TIPOS_AMBULANCIA,
        PARAMETROS.QUANTIDADE_MAXIMA_POR_TIPO,
        PARAMETROS.PARAMETRO_ALPHA,
        gerador_aleatorio,
    )

    # for id_regiao, coberturas_por_tipo in pontos_cobertos.items():
    #     for tipo, pontos in coberturas_por_tipo.items():
    #         print(f"P_{id_regiao}^{tipo} = {sorted(pontos)}")

    # return dataframe, pontos_cobertos
    print(verificar_viabilidade(solucao, PARAMETROS.QUANTIDADE_MAXIMA_POR_TIPO))
    print(solucao)
    print(f'Quant. pontos fisicamente cobertos: {len(obter_pontos_cobertos_pela_solucao(solucao, pontos_cobertos))}')
    print(f'Funcao objetivo inicial: {calcular_funcao_objetivo(solucao, pre_calculo=pre_calculo_cobertura)}')
    
    solucao_com_busca_local, fo_busca_local = busca_local(
        solucao_inicial=solucao,
        pontos_cobertos=pontos_cobertos,
        regioes=regioes,
        tipos_ambulancia=PARAMETROS.TIPOS_AMBULANCIA,
        quantidade_maxima_por_tipo=PARAMETROS.QUANTIDADE_MAXIMA_POR_TIPO,
        max_iteracoes_sem_melhora=PARAMETROS.MAX_ITERACOES_SEM_MELHORA,
        pre_calculo=pre_calculo_cobertura,
    )
    print(solucao_com_busca_local)
    print(f"Funcao objetivo busca local: {fo_busca_local}")
    print(f'Quant. pontos fisicamente cobertos: {len(obter_pontos_cobertos_pela_solucao(solucao_com_busca_local, pontos_cobertos))}')


    # 2 - validar separadamente construcao e busca local
    # 3 - validação do grasp
    # (obs.:) os códigos dos componentes (construcao, busca local, grasp) podem ser escritos em arquivos separados e importados aqui)

if __name__ == "__main__":
    main()
