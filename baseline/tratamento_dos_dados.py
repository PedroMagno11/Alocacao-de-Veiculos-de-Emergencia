import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame

RAIZ_PROJETO = Path(__file__).resolve().parents[1]
if str(RAIZ_PROJETO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROJETO))

from baseline.POSSIBLE import DISTRICTS_POINTS
from config.PARAMETROS import TIPOS_AMBULANCIA
from instances.read_instance import calcular_distancia_haversine
from instances.read_instance import obter_nome_coluna_cobertura


ARQUIVO_CSV = "instancia.csv"


def corrigir_texto_milhar(valor):
    """Converte pesos com separador de milhar para float."""
    if isinstance(valor, (int, float)):
        return float(valor)

    texto = str(valor).strip()
    if "." in texto:
        partes = texto.split(".")
        if len(partes[-1]) == 3 and texto.count(".") == 1 and "," not in texto:
            texto = texto.replace(".", "")
        elif texto.count(".") > 1:
            texto = "".join(partes[:-1]) + "." + partes[-1]

    try:
        return float(texto)
    except ValueError:
        return np.nan


def construir_dataframe_localidades(pontos: dict) -> DataFrame:
    """
    Monta o dataframe base com uma linha por localidade.

    O dicionario original usa a ordem [longitude, latitude, peso].
    """
    registros = [
        {
            "local_id": int(local_id),
            "longitude": longitude,
            "latitude": latitude,
            "peso": corrigir_texto_milhar(peso),
        }
        for local_id, (longitude, latitude, peso) in pontos.items()
    ]

    dataframe = pd.DataFrame(registros).sort_values("local_id").reset_index(drop=True)

    dataframe.loc[dataframe["peso"] > 10000, "peso"] /= 1000
    dataframe.loc[dataframe["peso"] > 1000, "peso"] /= 100
    dataframe.loc[dataframe["peso"] > 100, "peso"] /= 10

    return dataframe


def construir_matriz_distancias(dataframe: DataFrame) -> dict:
    """Calcula a distancia, em km, entre todas as localidades."""
    localidades = dataframe[["local_id", "latitude", "longitude"]].to_dict("records")
    matriz_distancias = {}

    for origem in localidades:
        id_origem = int(origem["local_id"])
        matriz_distancias[id_origem] = {}

        for destino in localidades:
            id_destino = int(destino["local_id"])
            matriz_distancias[id_origem][id_destino] = calcular_distancia_haversine(
                origem["latitude"],
                origem["longitude"],
                destino["latitude"],
                destino["longitude"],
            )

    return matriz_distancias


def obter_pontos_cobertos_por_tipo(
    matriz_de_distancias: dict,
    tipos_ambulancia: dict,
):
    """
    Retorna os pontos cobertos por cada localidade de origem para cada tipo.

    Estrutura:
      pontos_cobertos[tipo][local_id_origem] = [local_ids cobertos]
    """
    pontos_cobertos = {}

    for tipo, configuracao in tipos_ambulancia.items():
        raio_km = configuracao["raio_cobertura_km"]
        pontos_cobertos[tipo] = {}

        for origem, destinos in matriz_de_distancias.items():
            pontos_cobertos[tipo][origem] = [
                int(destino)
                for destino, distancia in destinos.items()
                if distancia <= raio_km
            ]

    return pontos_cobertos


def adicionar_pontos_cobertos_ao_dataframe(
    dataframe: DataFrame,
    pontos_cobertos: dict,
):
    """Adiciona uma coluna de cobertura para cada tipo de veiculo."""
    dataframe_com_cobertura = dataframe.copy()

    for tipo, cobertura_por_origem in pontos_cobertos.items():
        nome_da_coluna = obter_nome_coluna_cobertura(tipo)

        dataframe_com_cobertura[nome_da_coluna] = dataframe_com_cobertura[
            "local_id"
        ].map(lambda local_id: cobertura_por_origem.get(int(local_id), []))

    return dataframe_com_cobertura


def gerar_dados_com_cobertura():
    dataframe_localidades = construir_dataframe_localidades(DISTRICTS_POINTS)
    matriz_de_distancias = construir_matriz_distancias(dataframe_localidades)
    pontos_cobertos = obter_pontos_cobertos_por_tipo(
        matriz_de_distancias=matriz_de_distancias,
        tipos_ambulancia=TIPOS_AMBULANCIA,
    )

    return adicionar_pontos_cobertos_ao_dataframe(
        dataframe_localidades,
        pontos_cobertos,
    )


if __name__ == "__main__":
    df_com_pontos_cobertos = gerar_dados_com_cobertura()
    df_com_pontos_cobertos.to_csv(ARQUIVO_CSV, index=False)

    print(f"Arquivo gerado: {ARQUIVO_CSV}")
    print(f"Localidades: {len(df_com_pontos_cobertos)}")
    print(
        "Colunas de cobertura: "
        + ", ".join(
            coluna
            for coluna in df_com_pontos_cobertos.columns
            if coluna.startswith("pontos_cobertos_")
        )
    )
