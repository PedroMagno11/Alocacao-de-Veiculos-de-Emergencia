"""
Gerador de instancias aleatorias para o PMCS-FA.

As instancias geradas seguem o mesmo formato de instancia.csv:

    local_id, longitude, latitude, peso,
    pontos_cobertos_tipo_0, pontos_cobertos_tipo_1, ...

Exemplo:

    python3 gerar_instancias.py --quantidades 12 20 30
    python3 gerar_instancias.py --quantidades 50 --lat-min -23.35 --lat-max -23.25
"""

import argparse
import random
from pathlib import Path

import pandas

import config.PARAMETROS as PARAMETROS
from instances.read_instance import calcular_distancia_haversine, obter_nome_coluna_cobertura


CAMINHO_REFERENCIA_PADRAO = "instances/instancia.csv"
PASTA_SAIDA_PADRAO = "instances"


def obter_ranges_da_referencia(caminho_referencia):
    dataframe = pandas.read_csv(
        caminho_referencia,
        usecols=["latitude", "longitude", "peso"],
    )
    return {
        "lat_min": float(dataframe["latitude"].min()),
        "lat_max": float(dataframe["latitude"].max()),
        "lon_min": float(dataframe["longitude"].min()),
        "lon_max": float(dataframe["longitude"].max()),
        "peso_min": float(dataframe["peso"].min()),
        "peso_max": float(dataframe["peso"].max()),
    }


def limitar_valor(valor, minimo, maximo):
    return max(minimo, min(maximo, valor))


def deslocar_ponto(latitude, longitude, desvio_km, gerador):
    """
    Aplica um pequeno deslocamento aleatorio em km.

    A aproximacao abaixo e suficiente para gerar instancias sinteticas locais.
    """
    km_por_grau_latitude = 111.0
    km_por_grau_longitude = 102.0

    delta_latitude = gerador.gauss(0.0, desvio_km) / km_por_grau_latitude
    delta_longitude = gerador.gauss(0.0, desvio_km) / km_por_grau_longitude
    return latitude + delta_latitude, longitude + delta_longitude


def gerar_pontos(
    quantidade_pontos,
    lat_min,
    lat_max,
    lon_min,
    lon_max,
    peso_min,
    peso_max,
    quantidade_clusters,
    desvio_cluster_km,
    gerador,
):
    centros = [
        (
            gerador.uniform(lat_min, lat_max),
            gerador.uniform(lon_min, lon_max),
        )
        for _ in range(quantidade_clusters)
    ]

    linhas = []
    for local_id in range(quantidade_pontos):
        centro_latitude, centro_longitude = gerador.choice(centros)
        latitude, longitude = deslocar_ponto(
            centro_latitude,
            centro_longitude,
            desvio_cluster_km,
            gerador,
        )

        linhas.append(
            {
                "local_id": local_id,
                "longitude": limitar_valor(longitude, lon_min, lon_max),
                "latitude": limitar_valor(latitude, lat_min, lat_max),
                "peso": round(gerador.uniform(peso_min, peso_max), 2),
            }
        )

    return pandas.DataFrame(linhas)


def adicionar_colunas_de_cobertura(dataframe, tipos_ambulancia):
    coordenadas = dataframe[["local_id", "latitude", "longitude"]].values

    for tipo, configuracao in tipos_ambulancia.items():
        raio_cobertura = configuracao["raio_cobertura_km"]
        nome_coluna = obter_nome_coluna_cobertura(tipo)
        coberturas = []

        for linha_base in coordenadas:
            id_base = int(linha_base[0])
            latitude_base = float(linha_base[1])
            longitude_base = float(linha_base[2])
            pontos_cobertos = []

            for linha_demanda in coordenadas:
                id_demanda = int(linha_demanda[0])
                distancia = calcular_distancia_haversine(
                    latitude_base,
                    longitude_base,
                    float(linha_demanda[1]),
                    float(linha_demanda[2]),
                )
                if distancia <= raio_cobertura:
                    pontos_cobertos.append(id_demanda)

            if id_base not in pontos_cobertos:
                pontos_cobertos.append(id_base)

            coberturas.append(sorted(pontos_cobertos))

        dataframe[nome_coluna] = coberturas

    return dataframe


def gerar_instancia(
    quantidade_pontos,
    indice_instancia,
    argumentos,
    ranges,
    quantidade_clusters,
    desvio_cluster_km,
):
    gerador = random.Random(argumentos.seed + indice_instancia)
    dataframe = gerar_pontos(
        quantidade_pontos=quantidade_pontos,
        lat_min=argumentos.lat_min if argumentos.lat_min is not None else ranges["lat_min"],
        lat_max=argumentos.lat_max if argumentos.lat_max is not None else ranges["lat_max"],
        lon_min=argumentos.lon_min if argumentos.lon_min is not None else ranges["lon_min"],
        lon_max=argumentos.lon_max if argumentos.lon_max is not None else ranges["lon_max"],
        peso_min=argumentos.peso_min if argumentos.peso_min is not None else ranges["peso_min"],
        peso_max=argumentos.peso_max if argumentos.peso_max is not None else ranges["peso_max"],
        quantidade_clusters=quantidade_clusters,
        desvio_cluster_km=desvio_cluster_km,
        gerador=gerador,
    )
    return adicionar_colunas_de_cobertura(
        dataframe,
        PARAMETROS.TIPOS_AMBULANCIA,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Gera instancias aleatorias para validar o PMCS-FA."
    )
    parser.add_argument(
        "--quantidades",
        nargs="+",
        type=int,
        default=[12, 20, 30],
        help="Quantidade de pontos em cada instancia gerada.",
    )
    parser.add_argument("--saida", default=PASTA_SAIDA_PADRAO)
    parser.add_argument("--prefixo", default="instancia_aleatoria")
    parser.add_argument("--referencia", default=CAMINHO_REFERENCIA_PADRAO)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--clusters",
        nargs="+",
        type=int,
        default=[3],
        help="Quantidade(s) de clusters. Ex: --clusters 3 5 10",
    )
    parser.add_argument(
        "--desvios-cluster-km",
        nargs="+",
        type=float,
        default=[2.0],
        help="Desvio(s) dos clusters em km. Ex: --desvios-cluster-km 1.0 2.0 5.0",
    )
    parser.add_argument("--lat-min", type=float)
    parser.add_argument("--lat-max", type=float)
    parser.add_argument("--lon-min", type=float)
    parser.add_argument("--lon-max", type=float)
    parser.add_argument("--peso-min", type=float)
    parser.add_argument("--peso-max", type=float)
    return parser.parse_args()


def main():
    argumentos = parse_args()
    ranges = obter_ranges_da_referencia(argumentos.referencia)
    pasta_saida = Path(argumentos.saida)
    pasta_saida.mkdir(parents=True, exist_ok=True)

    indice_instancia = 1

    for quantidade_pontos in argumentos.quantidades:
        for quantidade_clusters in argumentos.clusters:
            for desvio_cluster_km in argumentos.desvios_cluster_km:
                dataframe = gerar_instancia(
                    quantidade_pontos=quantidade_pontos,
                    indice_instancia=indice_instancia,
                    argumentos=argumentos,
                    ranges=ranges,
                    quantidade_clusters=quantidade_clusters,
                    desvio_cluster_km=desvio_cluster_km,
                )

                desvio_formatado = str(desvio_cluster_km).replace(".", "_")

                pasta_instancia = (
                    pasta_saida
                    / f"{quantidade_pontos}p"
                    / f"{quantidade_clusters}_clusters"
                    / f"desvio_{desvio_formatado}km"
                )
                pasta_instancia.mkdir(parents=True, exist_ok=True)

                caminho_saida = pasta_instancia / (
                    f"{argumentos.prefixo}_{indice_instancia:02d}_"
                    f"{quantidade_pontos}p_"
                    f"{quantidade_clusters}c_"
                    f"desvio_{desvio_formatado}km.csv"
                )

                dataframe.to_csv(caminho_saida, index=False)
                print(f"Gerada: {caminho_saida}")

                indice_instancia += 1


if __name__ == "__main__":
    main()
