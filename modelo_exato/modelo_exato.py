"""
Modelo exato em Gurobi para o PMCS-FA.

Formulação implementada conforme a definição do problema no PDF:

    max sum |P_i^t| x_i^t
        - sum |P_i^t inter P_j^t'| y_ij^tt'

sujeito a:
    sum_i x_i^t <= N^t                 para todo tipo t
    sum_t x_i^t <= 1                   para toda região i
    y_ij^tt' = x_i^t * x_j^t'          por linearização
    x, y binários
"""

import gurobipy as gp
from gurobipy import GRB

import config.PARAMETROS as PARAMETROS
from instances.read_instance import ler_instancia, pre_computar_pontos_cobertos

CAMINHO_INSTANCIA = "instances/instancia_aleatoria_01_50p.csv"
# CAMINHO_INSTANCIA = "instancia.csv"


def carregar_instancia(caminho_instancia, tipos_ambulancia):
    dataframe = ler_instancia(caminho_instancia)
    regioes = [int(id_regiao) for id_regiao in dataframe["local_id"].tolist()]
    pontos_cobertos = pre_computar_pontos_cobertos(
        dataframe,
        tipos_ambulancia,
    )
    return dataframe, regioes, pontos_cobertos


def criar_indices_y_com_intersecao_positiva(
    regioes,
    tipos_ambulancia,
    pontos_cobertos,
):
    """
    Retorna os índices de y e os coeficientes de sobreposição.

    O PDF define y para todo par com j != i. Aqui criamos y apenas quando
    |P_i^t inter P_j^t'| > 0, pois pares com interseção zero teriam
    coeficiente zero no objetivo e não mudam a solução.
    """
    indices_y = []
    coeficiente_sobreposicao = {}

    for i in regioes:
        for t in tipos_ambulancia:
            pontos_i_t = pontos_cobertos[i][t]

            for j in regioes:
                if i == j:
                    continue

                for t_linha in tipos_ambulancia:
                    intersecao = len(pontos_i_t & pontos_cobertos[j][t_linha])
                    if intersecao == 0:
                        continue

                    indice = (i, t, j, t_linha)
                    indices_y.append(indice)
                    coeficiente_sobreposicao[indice] = intersecao

    return indices_y, coeficiente_sobreposicao


def construir_modelo_pmcs_fa(
    regioes,
    tipos_ambulancia,
    quantidade_maxima_por_tipo,
    pontos_cobertos,
):
    modelo = gp.Model("PMCS-FA")

    indices_x = [(i, t) for i in regioes for t in tipos_ambulancia]

    indices_y, coeficiente_sobreposicao = criar_indices_y_com_intersecao_positiva(
        regioes,
        tipos_ambulancia,
        pontos_cobertos,
    )

    # print(f"Total de variáveis x: {len(indices_x)}")
    # print(f"Total de variáveis y (com interseção positiva): {len(indices_y)}")

    # print(f"Indices x de exemplo: {indices_x}")
    # print(f"Indices y de exemplo: {indices_y}")

    x = modelo.addVars(indices_x, vtype=GRB.BINARY, name="x")
    y = modelo.addVars(indices_y, vtype=GRB.BINARY, name="y")

    cobertura_individual = gp.quicksum(
        len(pontos_cobertos[i][t]) * x[i, t] for i, t in indices_x
    )

    penalizacao_sobreposicao = gp.quicksum(
        coeficiente_sobreposicao[i, t, j, t_linha] * y[i, t, j, t_linha]
        for i, t, j, t_linha in indices_y
    )

    modelo.setObjective(
        cobertura_individual - penalizacao_sobreposicao,
        GRB.MAXIMIZE,
    )

    for t in tipos_ambulancia:
        modelo.addConstr(
            gp.quicksum(x[i, t] for i in regioes) <= quantidade_maxima_por_tipo[t],
            name=f"limite_frota_tipo_{t}",
        )

    for i in regioes:
        modelo.addConstr(
            gp.quicksum(x[i, t] for t in tipos_ambulancia) <= 1,
            name=f"uma_ambulancia_regiao_{i}",
        )

    for i, t, j, t_linha in indices_y:
        modelo.addConstr(y[i, t, j, t_linha] <= x[i, t])
        modelo.addConstr(y[i, t, j, t_linha] <= x[j, t_linha])
        modelo.addConstr(y[i, t, j, t_linha] >= x[i, t] + x[j, t_linha] - 1)

    modelo.update()
    return modelo, x, y


def imprimir_solucao(modelo, x, regioes, tipos_ambulancia):
    if modelo.Status != GRB.OPTIMAL:
        print(f"Nenhuma solução ótima encontrada. Status Gurobi: {modelo.Status}")
        return

    print(f"Função objetivo: {modelo.ObjVal:.2f}")
    print("Alocações escolhidas:")

    for i in regioes:
        for t in tipos_ambulancia:
            if x[i, t].X > 0.5:
                nome_tipo = tipos_ambulancia[t]["nome"]
                print(f"  região {i} -> tipo {t} ({nome_tipo})")


def main():
    dataframe, regioes, pontos_cobertos = carregar_instancia(
        CAMINHO_INSTANCIA,
        PARAMETROS.TIPOS_AMBULANCIA,
    )

    print("Instância carregada.")
    print(f"Pontos/regiões: {len(dataframe)}")
    print(f"Tipos de ambulância: {list(PARAMETROS.TIPOS_AMBULANCIA)}")

    modelo, x, _ = construir_modelo_pmcs_fa(
        regioes=regioes,
        tipos_ambulancia=PARAMETROS.TIPOS_AMBULANCIA,
        quantidade_maxima_por_tipo=PARAMETROS.QUANTIDADE_MAXIMA_POR_TIPO,
        pontos_cobertos=pontos_cobertos,
    )

    modelo.optimize()
    imprimir_solucao(
        modelo,
        x,
        regioes,
        PARAMETROS.TIPOS_AMBULANCIA,
    )


if __name__ == "__main__":
    main()
