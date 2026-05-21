import pandas as pd
from pandas import DataFrame
import numpy as np

from baseline.POSSIBLE import DISTRICTS_POINTS
from models import Localizacao, Alocacao
from util import distancia_haversine, distancia_euclidiana


def corrigir_texto_milhar(valor):
    # Se já for um número (int ou float), não precisa de tratamento de texto
    if isinstance(valor, (int, float)):
        return float(valor)
        
    texto = str(valor).strip()
    if '.' in texto:
        partes = texto.split('.')
        # Remove o ponto de um separador de milhar (ex: "5.432")
        if len(partes[-1]) == 3 and texto.count('.') == 1 and not ',' in texto:
            texto = texto.replace('.', '')
        elif texto.count('.') > 1:
            texto = "".join(partes[:-1]) + "." + partes[-1]
            
    try:
        return float(texto)
    except ValueError:
        # Se o texto for inválido, retornar nulo
        return np.nan

# 1. Cria o DataFrame direto do dicionário original
df_localizacoes = pd.DataFrame.from_dict(DISTRICTS_POINTS, orient='index')
df_localizacoes.columns = ['longitude', 'latitude', 'peso']

# 2. Transforma a coluna peso garantindo a conversão numérica limpa
df_localizacoes['peso'] = df_localizacoes['peso'].apply(corrigir_texto_milhar)

# 3. Aplica as regras de escala em ordem decrescente direto no DataFrame (Mais rápido)
df_localizacoes.loc[df_localizacoes['peso'] > 10000, 'peso'] /= 1000
df_localizacoes.loc[df_localizacoes['peso'] > 1000, 'peso'] /= 100
df_localizacoes.loc[df_localizacoes['peso'] > 100, 'peso'] /= 10


# gera locais de atendimento
locais_de_atendimento:list[Localizacao] = []

for i in range(len(df_localizacoes)):
    loc = Localizacao(**df_localizacoes.iloc[i].to_dict())
    locais_de_atendimento.append(loc)

alocacoes = []

for local_id, linha in df_localizacoes.iterrows():
    loc = Localizacao(**linha.to_dict())

    alocacao = Alocacao(
        local_id = str(local_id),
        latitude = loc.latitude,
        longitude = loc.longitude,
        peso = loc.peso)
    
    alocacoes.append(alocacao)

df_alocacoes = pd.DataFrame(alocacoes)
df_alocacoes = df_alocacoes.set_index("local_id")


def construir_matriz_distancias() -> dict:
    ids = list(DISTRICTS_POINTS.keys())
    matriz_distancias = {}

    for i in ids:
        matriz_distancias[i] = {}

        lng1, lat1, _ = DISTRICTS_POINTS[i]

        for j in ids:
            lng2, lat2, _ = DISTRICTS_POINTS[j]

            distancia = distancia_euclidiana(lat1, lng1, lat2, lng2)

            matriz_distancias[i][j] = distancia
    
    return matriz_distancias

def obter_regioes_cobertas_por_tempo(matriz_de_distancias: dict, tempos_em_minutos: list[float], velocidade_media_kmh: float = 40):
    areas_cobertas: dict = {}
    for tempo in tempos_em_minutos:
        raio_km = velocidade_media_kmh * (tempo / 60)

        areas_cobertas[str(tempo)] = {}

        for origem, destinos in matriz_de_distancias.items():
            areas_cobertas[str(tempo)][origem] = [
                destino
                for destino, distancia in destinos.items()
                if distancia <= raio_km 
            ]

    return areas_cobertas

def adicionar_areas_cobertas_ao_dataframe(df: DataFrame, areas_cobertas: dict):
    df_com_areas_cobertas = df.copy()

    df_com_areas_cobertas["id"] = df_com_areas_cobertas.index.astype(str)

    for tempo, area_coberta_partindo_da_origem in areas_cobertas.items():
        nome_da_coluna = f"cobertura_{tempo}_min"

        df_com_areas_cobertas[nome_da_coluna] = df_com_areas_cobertas["id"].map(
            lambda id_local: area_coberta_partindo_da_origem.get(id_local, [])
        )

    return df_com_areas_cobertas

# Gera a matriz de distância entre as regiões e o conjunto de regiões cobertas
VELOCIDADE_MEDIA_KMH = 40
TEMPOS_MIN = [5,10] # em minutos

matriz_de_distancias = construir_matriz_distancias()

areas_cobertas = obter_regioes_cobertas_por_tempo(
    matriz_de_distancias=matriz_de_distancias,
    velocidade_media_kmh=VELOCIDADE_MEDIA_KMH,
    tempos_em_minutos=TEMPOS_MIN
)

df_com_regioes_cobertas = adicionar_areas_cobertas_ao_dataframe(df_alocacoes, areas_cobertas=areas_cobertas)

df_com_regioes_cobertas.to_pickle("dados.pkl")
df_com_regioes_cobertas.to_csv("dados.csv")
