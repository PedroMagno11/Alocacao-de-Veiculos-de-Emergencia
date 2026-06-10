"""
Modelo exato em Gurobi para o PMCS-FA.

"""

import argparse
import csv
import time
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB

from config import PARAMETROS
from instances.read_instance import ler_instancia, pre_computar_pontos_cobertos


CAMINHO_INSTANCIA_PADRAO = (
    "instances/100p/8_clusters/desvio_1_0km/instancia_aleatoria_01_100p_8c_desvio_1_0km.csv"
)

PASTA_RESULTADOS_PADRAO = Path("results")


def carregar_instancia(caminho_instancia, tipos_ambulancia):
    dataframe = ler_instancia(caminho_instancia)

    regioes = [
        int(id_regiao)
        for id_regiao in dataframe["local_id"].tolist()
    ]

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
    indices_y = []
    coeficiente_sobreposicao = {}

    for i in regioes:
        for t in tipos_ambulancia:
            pontos_i_t = pontos_cobertos[i][t]

            for j in regioes:
                if i == j:
                    continue

                for t_linha in tipos_ambulancia:
                    intersecao = len(
                        pontos_i_t & pontos_cobertos[j][t_linha]
                    )

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

    indices_x = [
        (i, t)
        for i in regioes
        for t in tipos_ambulancia
    ]

    indices_y, coeficiente_sobreposicao = criar_indices_y_com_intersecao_positiva(
        regioes,
        tipos_ambulancia,
        pontos_cobertos,
    )

    print(f"Total de variáveis x: {len(indices_x)}")
    print(f"Total de variáveis y com interseção positiva: {len(indices_y)}")

    x = modelo.addVars(
        indices_x,
        vtype=GRB.BINARY,
        name="x",
    )

    y = modelo.addVars(
        indices_y,
        vtype=GRB.BINARY,
        name="y",
    )

    cobertura_individual = gp.quicksum(
        len(pontos_cobertos[i][t]) * x[i, t]
        for i, t in indices_x
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
            gp.quicksum(x[i, t] for i in regioes)
            <= quantidade_maxima_por_tipo[t],
            name=f"limite_frota_tipo_{t}",
        )

    for i in regioes:
        modelo.addConstr(
            gp.quicksum(x[i, t] for t in tipos_ambulancia) <= 1,
            name=f"uma_ambulancia_regiao_{i}",
        )

    for i, t, j, t_linha in indices_y:
        modelo.addConstr(
            y[i, t, j, t_linha] <= x[i, t],
            name=f"lin_y_le_x1_{i}_{t}_{j}_{t_linha}",
        )
        modelo.addConstr(
            y[i, t, j, t_linha] <= x[j, t_linha],
            name=f"lin_y_le_x2_{i}_{t}_{j}_{t_linha}",
        )
        modelo.addConstr(
            y[i, t, j, t_linha] >= x[i, t] + x[j, t_linha] - 1,
            name=f"lin_y_ge_sum_{i}_{t}_{j}_{t_linha}",
        )

    modelo.update()

    print(f"Total de variáveis no modelo: {modelo.NumVars}")
    print(f"Total de restrições no modelo: {modelo.NumConstrs}")

    return modelo, x, y


def obter_nome_status_gurobi(status):
    mapa_status = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.NUMERIC: "NUMERIC",
    }

    return mapa_status.get(status, str(status))


def obter_pasta_resultado_exato(
    caminho_instancia,
    pasta_resultados=PASTA_RESULTADOS_PADRAO,
):
    caminho = Path(caminho_instancia)

    try:
        relativo = caminho.relative_to("instances")
        pasta_relativa = relativo.parent
    except ValueError:
        pasta_relativa = Path("manual")

    pasta_saida = (
        Path(pasta_resultados)
        / "exato"
        / pasta_relativa
        / caminho.stem
    )

    pasta_saida.mkdir(parents=True, exist_ok=True)

    return pasta_saida


def salvar_resultado_exato(
    pasta_saida,
    caminho_instancia,
    modelo,
    tempo_execucao,
    x,
    regioes,
    tipos_ambulancia,
):
    pasta_saida = Path(pasta_saida)
    pasta_saida.mkdir(parents=True, exist_ok=True)

    status_nome = obter_nome_status_gurobi(modelo.Status)

    caminho_csv = pasta_saida / "resultado_exato.csv"
    caminho_solucao = pasta_saida / "solucao_exato.csv"

    possui_solucao = modelo.SolCount > 0

    with open(caminho_csv, "w", newline="", encoding="utf-8") as arquivo:
        writer = csv.DictWriter(
            arquivo,
            fieldnames=[
                "instancia",
                "status",
                "fo",
                "best_bound",
                "mip_gap",
                "tempo_s",
                "num_vars",
                "num_constrs",
                "num_solutions",
            ],
        )

        writer.writeheader()
        writer.writerow({
            "instancia": str(caminho_instancia),
            "status": status_nome,
            "fo": f"{modelo.ObjVal:.4f}" if possui_solucao else "",
            "best_bound": f"{modelo.ObjBound:.4f}" if possui_solucao else "",
            "mip_gap": f"{modelo.MIPGap:.8f}" if possui_solucao else "",
            "tempo_s": f"{tempo_execucao:.4f}",
            "num_vars": modelo.NumVars,
            "num_constrs": modelo.NumConstrs,
            "num_solutions": modelo.SolCount,
        })

    with open(caminho_solucao, "w", newline="", encoding="utf-8") as arquivo:
        writer = csv.DictWriter(
            arquivo,
            fieldnames=[
                "regiao",
                "tipo",
                "nome_tipo",
            ],
        )

        writer.writeheader()

        if possui_solucao:
            for i in regioes:
                for t in tipos_ambulancia:
                    if x[i, t].X > 0.5:
                        writer.writerow({
                            "regiao": i,
                            "tipo": t,
                            "nome_tipo": tipos_ambulancia[t]["nome"],
                        })

    try:
        modelo.write(str(pasta_saida / "modelo.lp"))
    except Exception as erro:
        print(f"[AVISO] Não foi possível salvar modelo.lp: {erro}")

    print(f"Resultado salvo em: {pasta_saida}")
    print(f"  - {caminho_csv}")
    print(f"  - {caminho_solucao}")


def imprimir_solucao(modelo, x, regioes, tipos_ambulancia):
    status_nome = obter_nome_status_gurobi(modelo.Status)

    if modelo.SolCount == 0:
        print(f"Nenhuma solução encontrada. Status Gurobi: {status_nome}")
        return

    print(f"Status Gurobi: {status_nome}")
    print(f"Função objetivo: {modelo.ObjVal:.2f}")

    if modelo.Status != GRB.OPTIMAL:
        print(f"Melhor bound: {modelo.ObjBound:.2f}")
        print(f"MIPGap: {modelo.MIPGap:.6f}")

    print("Alocações escolhidas:")

    for i in regioes:
        for t in tipos_ambulancia:
            if x[i, t].X > 0.5:
                nome_tipo = tipos_ambulancia[t]["nome"]
                print(f"  região {i} -> tipo {t} ({nome_tipo})")


def main():
    parser = argparse.ArgumentParser(
        description="Executa o modelo exato do PMCS-FA usando Gurobi."
    )

    parser.add_argument(
        "--instancia",
        default=CAMINHO_INSTANCIA_PADRAO,
        help="Caminho do arquivo CSV da instância.",
    )

    parser.add_argument(
        "--tempo-limite",
        type=float,
        default=None,
        help="Tempo limite do Gurobi em segundos. Ex: 3600.",
    )

    parser.add_argument(
        "--mip-gap",
        type=float,
        default=None,
        help="Gap relativo aceitável. Ex: 0.01 para 1%%.",
    )

    parser.add_argument(
        "--sem-log",
        action="store_true",
        help="Desativa saída do log do Gurobi no terminal.",
    )

    args = parser.parse_args()

    caminho_instancia = args.instancia
    pasta_saida = obter_pasta_resultado_exato(caminho_instancia)

    print("Carregando instância...")
    dataframe, regioes, pontos_cobertos = carregar_instancia(
        caminho_instancia,
        PARAMETROS.TIPOS_AMBULANCIA,
    )

    print("Instância carregada.")
    print(f"Arquivo: {caminho_instancia}")
    print(f"Pontos/regiões: {len(dataframe)}")
    print(f"Tipos de ambulância: {list(PARAMETROS.TIPOS_AMBULANCIA)}")
    print(f"Pasta de saída: {pasta_saida}")

    print("\nConstruindo modelo...")
    modelo, x, _ = construir_modelo_pmcs_fa(
        regioes=regioes,
        tipos_ambulancia=PARAMETROS.TIPOS_AMBULANCIA,
        quantidade_maxima_por_tipo=PARAMETROS.QUANTIDADE_MAXIMA_POR_TIPO,
        pontos_cobertos=pontos_cobertos,
    )

    modelo.setParam("OutputFlag", 0 if args.sem_log else 1)
    modelo.setParam("LogFile", str(pasta_saida / "gurobi.log"))

    if args.tempo_limite is not None:
        modelo.setParam("TimeLimit", args.tempo_limite)

    if args.mip_gap is not None:
        modelo.setParam("MIPGap", args.mip_gap)

    print("\nIniciando otimização...")
    inicio = time.perf_counter()
    modelo.optimize()
    tempo_execucao = time.perf_counter() - inicio
    print(f"Otimização finalizada em {tempo_execucao:.2f}s.")

    imprimir_solucao(
        modelo,
        x,
        regioes,
        PARAMETROS.TIPOS_AMBULANCIA,
    )

    salvar_resultado_exato(
        pasta_saida=pasta_saida,
        caminho_instancia=caminho_instancia,
        modelo=modelo,
        tempo_execucao=tempo_execucao,
        x=x,
        regioes=regioes,
        tipos_ambulancia=PARAMETROS.TIPOS_AMBULANCIA,
    )


if __name__ == "__main__":
    main()
