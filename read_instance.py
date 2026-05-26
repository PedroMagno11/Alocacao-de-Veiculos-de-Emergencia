"""
Leitura de instancias e pre-processamento dos conjuntos de cobertura P_i^t
para o PMCS-FA.
"""

from math import asin, cos, radians, sin, sqrt

import pandas


def ler_instancia(caminho_arquivo):
    """
    Le uma instancia em CSV e valida as colunas minimas esperadas.

    Colunas obrigatorias:
      - local_id
      - latitude
      - longitude
    """
    dataframe = pandas.read_csv(caminho_arquivo)
    colunas_obrigatorias = {"local_id", "latitude", "longitude"}
    colunas_faltantes = colunas_obrigatorias - set(dataframe.columns)

    if colunas_faltantes:
        faltantes = ", ".join(sorted(colunas_faltantes))
        raise ValueError(f"Instancia invalida. Colunas faltantes: {faltantes}")

    return dataframe


def calcular_distancia_haversine(latitude1, longitude1, latitude2, longitude2):
    """
    Retorna a distancia em quilometros entre dois pontos geograficos
    definidos por latitude e longitude em graus decimais.
    """
    raio_terra_km = 6371.0
    latitude1, longitude1, latitude2, longitude2 = map(
        radians, [latitude1, longitude1, latitude2, longitude2]
    )
    diferenca_latitude = latitude2 - latitude1
    diferenca_longitude = longitude2 - longitude1
    termo_haversine = (
        sin(diferenca_latitude / 2) ** 2
        + cos(latitude1) * cos(latitude2) * sin(diferenca_longitude / 2) ** 2
    )
    return 2.0 * raio_terra_km * asin(sqrt(termo_haversine))


def pre_computar_pontos_cobertos(dataframe, tipos_ambulancia):
    """
    Pre-computa, para cada regiao base e cada tipo de ambulancia, o conjunto
    de pontos de demanda cobertos caso aquela ambulancia seja alocada naquela
    regiao.

    Retorna:
      pontos_cobertos[id_regiao][tipo] = frozenset de local_ids cobertos
    """
    coordenadas = dataframe[["local_id", "latitude", "longitude"]].values
    pontos_cobertos = {}

    for linha_regiao in coordenadas:
        id_regiao = int(linha_regiao[0])
        latitude = linha_regiao[1]
        longitude = linha_regiao[2]
        pontos_cobertos[id_regiao] = {}

        for tipo, configuracao in tipos_ambulancia.items():
            raio = configuracao["raio_cobertura_km"]
            pontos_cobertos[id_regiao][tipo] = frozenset(
                int(linha_ponto[0])
                for linha_ponto in coordenadas
                if calcular_distancia_haversine(
                    latitude,
                    longitude,
                    linha_ponto[1],
                    linha_ponto[2],
                )
                <= raio
            )

    return pontos_cobertos
