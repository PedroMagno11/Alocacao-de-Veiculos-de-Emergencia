import random
from pandas import DataFrame


def calcular_score(df: DataFrame, solucao, TEMPO_COBERTURA):
    regioes_cobertas_total = set()

    score_total = 0.0
    score_ambulancia_do_tipo_A = 0.0
    score_ambulancia_do_tipo_B = 0.0

    for local_id, tipo_ambulancia in solucao:
        regioes_cobertas = set(df.loc[local_id, f"cobertura_{TEMPO_COBERTURA}_min"])

        novas_regioes = regioes_cobertas - regioes_cobertas_total

        if not novas_regioes:
            continue

        if tipo_ambulancia == "A":
            ganho = df.loc[list(novas_regioes), "demanda_complexa"].sum()
            score_ambulancia_do_tipo_A += ganho

        elif tipo_ambulancia == "B":
            ganho = df.loc[list(novas_regioes), "demanda_simples"].sum()
            score_ambulancia_do_tipo_B += ganho

        else:
            ganho = 0.0

        score_total += ganho
        regioes_cobertas_total.update(novas_regioes)

    # BUG CORRIGIDO: return estava dentro do for, retornava após a 1ª iteração
    return {
        "score_total": score_total,
        "score_ambulancia_tipo_A": score_ambulancia_do_tipo_A,
        "score_ambulancia_tipo_B": score_ambulancia_do_tipo_B,
        "quant_regioes_cobertas_total": len(regioes_cobertas_total)
    }

# ---------------------------------------------------------

def eh_melhor(nova_avaliacao, melhor_avaliacao) -> bool:
    if melhor_avaliacao is None:
        return True

    return nova_avaliacao["score_total"] > melhor_avaliacao["score_total"]

# ---------------------------------------------------------

def calcular_ganho_da_alocacao_de_ambulancia_para_regiao(df: DataFrame, local_id, tipo_ambulancia, regioes_ja_cobertas, TEMPO_DE_COBERTURA):
    regioes_cobertas = set(df.loc[local_id, f"cobertura_{TEMPO_DE_COBERTURA}_min"])

    novas_regioes = regioes_cobertas - regioes_ja_cobertas

    if not novas_regioes:
        return 0

    if tipo_ambulancia == "A":
        return df.loc[list(novas_regioes), "demanda_complexa"].sum()

    elif tipo_ambulancia == "B":
        return df.loc[list(novas_regioes), "demanda_simples"].sum()

    return 0

# ----------------------------------------------------------------------

def escolher_da_lista_de_candidatos_restritos(avaliacoes, ALPHA):
    avaliacoes = sorted(
        avaliacoes,
        key=lambda x: x["ganho"],
        reverse=True
    )

    melhor_ganho = avaliacoes[0]["ganho"]
    pior_ganho = avaliacoes[-1]["ganho"]

    limite = melhor_ganho - ALPHA * (melhor_ganho - pior_ganho)

    lista_restrita = [
        avaliacao
        for avaliacao in avaliacoes
        if avaliacao["ganho"] >= limite
    ]

    return random.choice(lista_restrita)

# --------------------------------------------------------------------

def construir_solucao(df: DataFrame, QTD_AMBULANCIAS_A, QTD_AMBULANCIAS_B, ALPHA, TEMPO_COBERTURA):
    solucao = []
    regioes_ja_cobertas = set()

    # Gera uma lista de ambulâncias com N ambulâncias do tipo A e K ambulâncias do tipo B
    ambulancias_para_alocar = (["A"] * QTD_AMBULANCIAS_A + ["B"] * QTD_AMBULANCIAS_B)

    # Reorganiza de maneira aleatória a lista de ambulâncias disponíveis para alocação.
    random.shuffle(ambulancias_para_alocar)

    locais_disponiveis = set(df.index.astype(str))

    for tipo_ambulancia in ambulancias_para_alocar:
        avaliacoes = []

        for local_id in locais_disponiveis:
            ganho = calcular_ganho_da_alocacao_de_ambulancia_para_regiao(
                df=df,
                local_id=local_id,
                tipo_ambulancia=tipo_ambulancia,
                regioes_ja_cobertas=regioes_ja_cobertas,
                TEMPO_DE_COBERTURA=TEMPO_COBERTURA
            )

            avaliacoes.append({
                "local_id": local_id,
                "tipo_ambulancia": tipo_ambulancia,
                "ganho": ganho
            })

        escolhido = escolher_da_lista_de_candidatos_restritos(avaliacoes=avaliacoes, ALPHA=ALPHA)

        local_escolhido = escolhido["local_id"]
        tipo_escolhido = escolhido["tipo_ambulancia"]

        solucao.append((local_escolhido, tipo_escolhido))

        regioes_cobertas = set(df.loc[local_escolhido, f"cobertura_{TEMPO_COBERTURA}_min"])
        regioes_ja_cobertas.update(regioes_cobertas)

        locais_disponiveis.remove(local_escolhido)

    # BUG CORRIGIDO: return estava dentro do for, retornava após alocar só 1 ambulância.
    # A solução é o conjunto completo de todas as alocações (A + B).
    return solucao

# ---------------------------------------------------------

def gerar_vizinhos(df: DataFrame, solucao):
    vizinhos = []

    locais_usados = {
        local_id
        for local_id, _ in solucao
    }

    candidatos_livres = [
        local_id
        for local_id in df.index.astype(str)
        if local_id not in locais_usados
    ]

    for indice, (local_atual, tipo_ambulancia) in enumerate(solucao):
        for novo_local in candidatos_livres:
            nova_solucao = solucao.copy()
            nova_solucao[indice] = (novo_local, tipo_ambulancia)
            vizinhos.append(nova_solucao)

    return vizinhos

# ---------------------------------------------------------

def busca_local(df: DataFrame, solucao, TEMPO_COBERTURA):
    melhor_solucao = solucao.copy()
    melhor_avaliacao = calcular_score(df, melhor_solucao, TEMPO_COBERTURA=TEMPO_COBERTURA)

    melhorou = True

    while melhorou:
        melhorou = False

        for vizinho in gerar_vizinhos(df=df, solucao=melhor_solucao):
            avaliacao_vizinho = calcular_score(df=df, solucao=vizinho, TEMPO_COBERTURA=TEMPO_COBERTURA)

            if eh_melhor(nova_avaliacao=avaliacao_vizinho, melhor_avaliacao=melhor_avaliacao):
                melhor_solucao = vizinho
                melhor_avaliacao = avaliacao_vizinho
                melhorou = True
                break

    return melhor_solucao

# ---------------------------------------------------------
