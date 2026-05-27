from read_instance import ler_instancia, pre_computar_pontos_cobertos
import PARAMETROS
from construtivo import construir_solucao
from avaliador_viabilidade import verificar_viabilidade
from busca_local import obter_pontos_cobertos_pela_solucao
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
    print(f'Quant. pontos cobertos: {len(obter_pontos_cobertos_pela_solucao(solucao, pontos_cobertos))}')
    
    # 2 - validar separadamente construcao e busca local
    # 3 - validação do grasp
    # (obs.:) os códigos dos componentes (construcao, busca local, grasp) podem ser escritos em arquivos separados e importados aqui)

if __name__ == "__main__":
    main()
