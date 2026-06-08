"""
Funções de visualização para o PMCS-FA.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Circle


# ============================================================
# AUXILIAR
# ============================================================

def _obter_dataframe_indexado(dataframe):
    return dataframe.set_index("local_id")


# ============================================================
# 1 - SOMENTE AS SOLUÇÕES
# ============================================================

def plotar_solucoes(
        dataframe,
        solucao_construtivo,
        solucao_busca_local):
    """
    Exibe apenas os pontos das soluções.

    Azul  -> Construtivo
    Roxo  -> Busca Local
    """

    df = _obter_dataframe_indexado(dataframe)

    plt.figure(figsize=(10, 8))

    for local_id, tipo in solucao_construtivo:
        linha = df.loc[local_id]

        plt.scatter(
            linha["longitude"],
            linha["latitude"],
            color="blue",
            s=150,
            marker="o"
        )

        plt.annotate(
            f"C{tipo}",
            (linha["longitude"], linha["latitude"])
        )

    for local_id, tipo in solucao_busca_local:
        linha = df.loc[local_id]

        plt.scatter(
            linha["longitude"],
            linha["latitude"],
            color="purple",
            s=180,
            marker="X"
        )

        plt.annotate(
            f"BL{tipo}",
            (linha["longitude"], linha["latitude"])
        )

    plt.title("Construtivo x Busca Local")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ============================================================
# 2 - MOVIMENTOS DA BUSCA LOCAL
# ============================================================

def plotar_movimentos_busca_local(
        dataframe,
        solucao_construtivo,
        solucao_busca_local):
    """
    Mostra o deslocamento realizado pela busca local.
    """

    df = _obter_dataframe_indexado(dataframe)

    plt.figure(figsize=(10, 8))

    construtivo = sorted(list(solucao_construtivo))
    busca_local = sorted(list(solucao_busca_local))

    quantidade = min(
        len(construtivo),
        len(busca_local)
    )

    for i in range(quantidade):

        id_c, _ = construtivo[i]
        id_b, _ = busca_local[i]

        ponto_c = df.loc[id_c]
        ponto_b = df.loc[id_b]

        plt.scatter(
            ponto_c["longitude"],
            ponto_c["latitude"],
            color="blue",
            s=120
        )

        plt.scatter(
            ponto_b["longitude"],
            ponto_b["latitude"],
            color="purple",
            s=180,
            marker="X"
        )

        plt.arrow(
            ponto_c["longitude"],
            ponto_c["latitude"],
            ponto_b["longitude"] - ponto_c["longitude"],
            ponto_b["latitude"] - ponto_c["latitude"],
            alpha=0.4,
            head_width=0.001,
            length_includes_head=True
        )

    plt.title("Movimentos da Busca Local")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ============================================================
# 3 - COBERTURA DAS AMBULÂNCIAS
# ============================================================

def plotar_cobertura(
        dataframe,
        solucao,
        tipos_ambulancia):
    """
    Exibe os círculos de cobertura.
    """

    df = _obter_dataframe_indexado(dataframe)

    plt.figure(figsize=(10, 8))

    for local_id, tipo in solucao:

        linha = df.loc[local_id]

        longitude = linha["longitude"]
        latitude = linha["latitude"]

        raio_km = tipos_ambulancia[tipo][
            "raio_cobertura_km"
        ]

        # Aproximação:
        raio_graus = raio_km / 111.0

        plt.scatter(
            longitude,
            latitude,
            s=150,
            color="blue"
        )

        circulo = Circle(
            (longitude, latitude),
            raio_graus,
            fill=False,
            alpha=0.3
        )

        plt.gca().add_patch(circulo)

        plt.annotate(
            f"T{tipo}",
            (longitude, latitude)
        )

    plt.title("Cobertura das Ambulâncias")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ============================================================
# 4 - EVOLUÇÃO DO GRASP
# ============================================================

def plotar_evolucao_grasp(historico_fo):
    """
    Mostra a evolução da função objetivo.
    """

    plt.figure(figsize=(10, 5))

    plt.plot(
        range(1, len(historico_fo) + 1),
        historico_fo,
        marker="o"
    )

    plt.title("Evolução da Função Objetivo")
    plt.xlabel("Iteração")
    plt.ylabel("FO")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# ============================================================
# 5 - DISPERSÃO DAS SOLUÇÕES DO GRASP
# ============================================================

def plotar_dispersao_grasp(historico_fo):
    """
    Visualização estilo artigo científico.
    """

    plt.figure(figsize=(10, 5))

    plt.scatter(
        range(1, len(historico_fo) + 1),
        historico_fo,
        alpha=0.7
    )

    plt.title("Dispersão das Soluções do GRASP")
    plt.xlabel("Iteração")
    plt.ylabel("FO")

    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plotar_funcao_objetivo_por_iteracao(historico_fo):
    """
    Eixo X = Iteração
    Eixo Y = Função Objetivo
    """

    iteracoes = range(1, len(historico_fo) + 1)

    plt.figure(figsize=(12, 6))

    plt.scatter(
        iteracoes,
        historico_fo,
        s=50,          # tamanho dos pontos
        alpha=0.8
    )

    plt.title("Evolução da Função Objetivo")
    plt.xlabel("Iteração")
    plt.ylabel("Função Objetivo")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt


def plotar_comparacao_funcao_objetivo_antes_e_apos_busca_local(
        historico_fo_antes_busca_local_f1,
        historico_fo_pos_busca_local_f1,
        historico_fo_antes_busca_local_f2 = None,
        historico_fo_pos_busca_local_f2 = None,
        legenda_azul="Pré-Busca Local",
        legenda_roxo="Pós-Busca Local"):
    """
    Compara duas execuções.

    Eixo X = Iteração
    Eixo Y = Função Objetivo

    Azul -> historico_azul
    Roxo -> historico_roxo
    """

    iteracoes_fo_pre_busca_local = range(
        1,
        len(historico_fo_antes_busca_local_f1) + 1
    )

    iteracoes_fo_pos_busca_local = range(
        1,
        len(historico_fo_pos_busca_local_f1) + 1
    )

    if(historico_fo_antes_busca_local_f2 is not None and historico_fo_pos_busca_local_f2 is not None):
        iteracoes_fo_pre_busca_local_f2 = range(1, len(historico_fo_antes_busca_local_f2) + 1)
        iteracoes_fo_pos_busca_local_f2 = range(1, len(historico_fo_pos_busca_local_f2) + 1)
        
        plt.scatter(
        iteracoes_fo_pre_busca_local_f2,
        historico_fo_antes_busca_local_f2,
        color="red",
        alpha=0.8,
        s=40,
        label=legenda_azul
        )

        plt.scatter(
            iteracoes_fo_pos_busca_local_f2,
            historico_fo_pos_busca_local_f2,
            color="green",
            alpha=0.8,
            s=40,
            label=legenda_roxo
        )

    plt.figure(figsize=(12, 6))

    plt.scatter(
        iteracoes_fo_pre_busca_local,
        historico_fo_antes_busca_local_f1,
        color="blue",
        alpha=0.8,
        s=40,
        label=legenda_azul
    )

    plt.scatter(
        iteracoes_fo_pos_busca_local,
        historico_fo_pos_busca_local_f1,
        color="purple",
        alpha=0.8,
        s=40,
        label=legenda_roxo
    )

    plt.title("Comparação das Soluções")
    plt.xlabel("Iteração")
    plt.ylabel("Função Objetivo")

    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()