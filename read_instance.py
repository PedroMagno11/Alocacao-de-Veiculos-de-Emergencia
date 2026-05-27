"""
Leitura de instancias e consulta dos conjuntos de cobertura P_i^t
pre-calculados para o PMCS-FA.
"""

from ast import literal_eval
from math import asin, cos, radians, sin, sqrt

import pandas


PREFIXO_COLUNA_COBERTURA = "pontos_cobertos_tipo_"


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


def obter_nome_coluna_cobertura(tipo):
    """Retorna o nome da coluna que armazena a cobertura de um tipo."""
    return f"{PREFIXO_COLUNA_COBERTURA}{tipo}"


def _normalizar_pontos_cobertos(valor):
    """
    Converte a cobertura armazenada no dataframe para frozenset de inteiros.

    A conversao aceita listas/sets ja em memoria e tambem strings no formato
    "[1, 2, 3]", comuns quando um dataframe pre-processado foi salvo em CSV.
    """
    if isinstance(valor, frozenset):
        return valor
    if isinstance(valor, str):
        valor = literal_eval(valor)
    return frozenset(int(ponto) for ponto in valor)


def dataframe_tem_pontos_cobertos(dataframe, tipos_ambulancia):
    """Verifica se o dataframe ja possui as colunas de cobertura esperadas."""
    return all(
        obter_nome_coluna_cobertura(tipo) in dataframe.columns
        for tipo in tipos_ambulancia
    )


def obter_colunas_cobertura_faltantes(dataframe, tipos_ambulancia):
    """Retorna as colunas de cobertura que nao existem no dataframe."""
    return [
        obter_nome_coluna_cobertura(tipo)
        for tipo in tipos_ambulancia
        if obter_nome_coluna_cobertura(tipo) not in dataframe.columns
    ]


def obter_pontos_cobertos_do_dataframe(dataframe, tipos_ambulancia):
    """
    Consulta as colunas de cobertura do dataframe e retorna a estrutura usada
    pelo algoritmo:

      pontos_cobertos[id_regiao][tipo] = frozenset de local_ids cobertos
    """
    pontos_cobertos = {}

    for _, linha in dataframe.iterrows():
        id_regiao = int(linha["local_id"])
        pontos_cobertos[id_regiao] = {}

        for tipo in tipos_ambulancia:
            nome_coluna = obter_nome_coluna_cobertura(tipo)
            pontos_cobertos[id_regiao][tipo] = _normalizar_pontos_cobertos(
                linha[nome_coluna]
            )

    return pontos_cobertos


def pre_computar_pontos_cobertos(
    dataframe,
    tipos_ambulancia,
):
    """
    Le do dataframe, para cada regiao base e cada tipo de ambulancia, o conjunto
    de pontos de demanda cobertos.

    As coberturas devem estar pre-calculadas em uma coluna por tipo:
      pontos_cobertos_tipo_<tipo>

    Retorna:
      pontos_cobertos[id_regiao][tipo] = frozenset de local_ids cobertos
    """
    colunas_faltantes = obter_colunas_cobertura_faltantes(
        dataframe,
        tipos_ambulancia,
    )
    if colunas_faltantes:
        faltantes = ", ".join(colunas_faltantes)
        raise ValueError(
            "Instancia sem pontos cobertos pre-calculados. "
            f"Colunas faltantes: {faltantes}. "
            "Execute baseline/tratamento_dos_dados.py para gerar a instancia."
        )

    return obter_pontos_cobertos_do_dataframe(dataframe, tipos_ambulancia)
