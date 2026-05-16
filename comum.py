import random
from pandas import DataFrame


# -----------------------------------------------------------------------
# Score completo — usado na construção e para a solução final
# -----------------------------------------------------------------------

def calcular_score(df: DataFrame, solucao, TEMPO_COBERTURA):
    regioes_cobertas_total = set()

    score_total            = 0.0
    score_ambulancia_tipo_A = 0.0
    score_ambulancia_tipo_B = 0.0

    for local_id, tipo_ambulancia in solucao:
        regioes_cobertas = set(df.loc[local_id, f"cobertura_{TEMPO_COBERTURA}_min"])
        novas_regioes    = regioes_cobertas - regioes_cobertas_total

        if not novas_regioes:
            continue

        if tipo_ambulancia == "A":
            ganho = df.loc[list(novas_regioes), "demanda_complexa"].sum()
            score_ambulancia_tipo_A += ganho
        elif tipo_ambulancia == "B":
            ganho = df.loc[list(novas_regioes), "demanda_simples"].sum()
            score_ambulancia_tipo_B += ganho
        else:
            ganho = 0.0

        score_total += ganho
        regioes_cobertas_total.update(novas_regioes)

    return {
        "score_total":                  score_total,
        "score_ambulancia_tipo_A":      score_ambulancia_tipo_A,
        "score_ambulancia_tipo_B":      score_ambulancia_tipo_B,
        "quant_regioes_cobertas_total": len(regioes_cobertas_total)
    }


# -----------------------------------------------------------------------
# Estado expandido — base para a avaliação incremental
# -----------------------------------------------------------------------

def calcular_estado(df: DataFrame, solucao, TEMPO_COBERTURA) -> dict:
    """
    Calcula o score E mantém, para cada posição da solução, quais regiões
    ela cobre de forma exclusiva (i.e. que só ela cobre e nenhuma outra
    posição cobre).

    Retorna
    -------
    {
        "score_total": float,
        "score_ambulancia_tipo_A": float,
        "score_ambulancia_tipo_B": float,
        "quant_regioes_cobertas_total": int,
        "regioes_cobertas_total": set,          ← todas as regiões cobertas
        "contribuicao": list[set],              ← regiões novas que cada posição trouxe
        "cobertura_por_regiao": dict[str, int], ← quantas ambulâncias cobrem cada região
    }

    `contribuicao[i]` = regiões que a posição i adicionou ao conjunto global.
    `cobertura_por_regiao` permite saber quais regiões ficam descobertas
    ao remover uma ambulância.
    """
    regioes_cobertas_total  = set()
    cobertura_por_regiao    = {}   # regiao_id -> contagem de ambulâncias que a cobrem
    contribuicao            = []   # contribuicao[i] = novas regiões trazidas pela posição i

    score_total             = 0.0
    score_ambulancia_tipo_A = 0.0
    score_ambulancia_tipo_B = 0.0

    for local_id, tipo_ambulancia in solucao:
        regioes_cobertas = set(df.loc[local_id, f"cobertura_{TEMPO_COBERTURA}_min"])
        novas_regioes    = regioes_cobertas - regioes_cobertas_total

        # Atualiza contagem de cobertura para TODAS as regiões deste local
        # (não só as novas), pois outras posições podem já cobri-las.
        for r in regioes_cobertas:
            cobertura_por_regiao[r] = cobertura_por_regiao.get(r, 0) + 1

        if novas_regioes:
            if tipo_ambulancia == "A":
                ganho = df.loc[list(novas_regioes), "demanda_complexa"].sum()
                score_ambulancia_tipo_A += ganho
            elif tipo_ambulancia == "B":
                ganho = df.loc[list(novas_regioes), "demanda_simples"].sum()
                score_ambulancia_tipo_B += ganho
            else:
                ganho = 0.0

            score_total += ganho
            regioes_cobertas_total.update(novas_regioes)

        contribuicao.append(novas_regioes)

    return {
        "score_total":                  score_total,
        "score_ambulancia_tipo_A":      score_ambulancia_tipo_A,
        "score_ambulancia_tipo_B":      score_ambulancia_tipo_B,
        "quant_regioes_cobertas_total": len(regioes_cobertas_total),
        "regioes_cobertas_total":       regioes_cobertas_total,
        "contribuicao":                 contribuicao,
        "cobertura_por_regiao":         cobertura_por_regiao,
    }


# -----------------------------------------------------------------------
# Delta incremental — custo de trocar uma posição sem recalcular tudo
# -----------------------------------------------------------------------

def calcular_delta(
    df: DataFrame,
    estado: dict,
    indice: int,
    local_antigo: str,
    novo_local: str,
    tipo_ambulancia: str,
    TEMPO_COBERTURA: int,
) -> float:
    """
    Calcula a variação de score (delta) ao substituir `local_antigo` por
    `novo_local` na posição `indice`, sem recalcular o score do zero.

    Estratégia
    ----------
    1. Regiões perdidas: as que `local_antigo` cobria exclusivamente
       (cobertura_por_regiao == 1). Ao removê-lo, essas regiões ficam
       descobertas → perda de ganho.

    2. Regiões ganhas: as que `novo_local` cobre e que ainda não estão
       cobertas por nenhuma outra posição → ganho adicional.

    Retorna o delta (pode ser negativo).
    """
    cobertura_por_regiao   = estado["cobertura_por_regiao"]
    regioes_cobertas_total = estado["regioes_cobertas_total"]

    col_demanda = "demanda_complexa" if tipo_ambulancia == "A" else "demanda_simples"

    # --- Perda: regiões que ficam descobertas ao remover local_antigo ----
    regioes_antigo    = set(df.loc[local_antigo, f"cobertura_{TEMPO_COBERTURA}_min"])
    regioes_exclusivas = {r for r in regioes_antigo if cobertura_por_regiao.get(r, 0) == 1}

    perda = df.loc[list(regioes_exclusivas), col_demanda].sum() if regioes_exclusivas else 0.0

    # --- Ganho: regiões novas trazidas por novo_local --------------------
    regioes_novo  = set(df.loc[novo_local, f"cobertura_{TEMPO_COBERTURA}_min"])
    # Regiões que novo_local cobre e que não estão cobertas por nenhuma
    # outra posição (descontando a que estamos removendo)
    regioes_descobertas = regioes_cobertas_total - regioes_antigo  # cobertas sem local_antigo
    regioes_novas       = regioes_novo - regioes_descobertas - regioes_exclusivas

    ganho = df.loc[list(regioes_novas), col_demanda].sum() if regioes_novas else 0.0

    return ganho - perda


# -----------------------------------------------------------------------

def eh_melhor(nova_avaliacao, melhor_avaliacao) -> bool:
    if melhor_avaliacao is None:
        return True
    return nova_avaliacao["score_total"] > melhor_avaliacao["score_total"]


# -----------------------------------------------------------------------

def calcular_ganho_da_alocacao_de_ambulancia_para_regiao(
    df: DataFrame, local_id, tipo_ambulancia, regioes_ja_cobertas, TEMPO_DE_COBERTURA
):
    regioes_cobertas = set(df.loc[local_id, f"cobertura_{TEMPO_DE_COBERTURA}_min"])
    novas_regioes    = regioes_cobertas - regioes_ja_cobertas

    if not novas_regioes:
        return 0

    if tipo_ambulancia == "A":
        return df.loc[list(novas_regioes), "demanda_complexa"].sum()
    elif tipo_ambulancia == "B":
        return df.loc[list(novas_regioes), "demanda_simples"].sum()

    return 0


# -----------------------------------------------------------------------

def escolher_da_lista_de_candidatos_restritos(avaliacoes, ALPHA):
    avaliacoes = sorted(avaliacoes, key=lambda x: x["ganho"], reverse=True)

    melhor_ganho = avaliacoes[0]["ganho"]
    pior_ganho   = avaliacoes[-1]["ganho"]
    limite       = melhor_ganho - ALPHA * (melhor_ganho - pior_ganho)

    lista_restrita = [a for a in avaliacoes if a["ganho"] >= limite]

    return random.choice(lista_restrita)


# -----------------------------------------------------------------------

def construir_solucao(df: DataFrame, QTD_AMBULANCIAS_A, QTD_AMBULANCIAS_B, ALPHA, TEMPO_COBERTURA):
    solucao             = []
    regioes_ja_cobertas = set()

    ambulancias_para_alocar = ["A"] * QTD_AMBULANCIAS_A + ["B"] * QTD_AMBULANCIAS_B
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
            avaliacoes.append({"local_id": local_id, "tipo_ambulancia": tipo_ambulancia, "ganho": ganho})

        escolhido      = escolher_da_lista_de_candidatos_restritos(avaliacoes=avaliacoes, ALPHA=ALPHA)
        local_escolhido = escolhido["local_id"]
        tipo_escolhido  = escolhido["tipo_ambulancia"]

        solucao.append((local_escolhido, tipo_escolhido))

        regioes_cobertas = set(df.loc[local_escolhido, f"cobertura_{TEMPO_COBERTURA}_min"])
        regioes_ja_cobertas.update(regioes_cobertas)
        locais_disponiveis.remove(local_escolhido)

    return solucao


# -----------------------------------------------------------------------
# Busca local com avaliação incremental
# -----------------------------------------------------------------------

def busca_local(df: DataFrame, solucao, TEMPO_COBERTURA, MAX_ROUNDS_SEM_MELHORIA: int = None):
    """
    Busca local com avaliação incremental (delta).

    Em vez de chamar calcular_score() para cada vizinho (~24.000 vezes
    por round com 3016 locais), calcula apenas o delta da troca proposta.
    Isso reduz o custo por round de O(n * regioes) para O(regioes_local).

    MAX_ROUNDS_SEM_MELHORIA
        None → para só quando nenhum vizinho melhora (convergência total)
        N    → para após N rounds consecutivos sem melhoria
    """
    melhor_solucao = solucao.copy()
    estado         = calcular_estado(df, melhor_solucao, TEMPO_COBERTURA)

    locais_usados      = {local_id for local_id, _ in melhor_solucao}
    candidatos_livres  = [lid for lid in df.index.astype(str) if lid not in locais_usados]

    rounds_sem_melhoria = 0

    while True:
        melhorou = False

        for indice, (local_atual, tipo_ambulancia) in enumerate(melhor_solucao):
            for novo_local in candidatos_livres:

                delta = calcular_delta(
                    df=df,
                    estado=estado,
                    indice=indice,
                    local_antigo=local_atual,
                    novo_local=novo_local,
                    tipo_ambulancia=tipo_ambulancia,
                    TEMPO_COBERTURA=TEMPO_COBERTURA,
                )

                if delta > 0:
                    # Aceita a troca e atualiza solução + candidatos
                    melhor_solucao[indice] = (novo_local, tipo_ambulancia)

                    candidatos_livres.remove(novo_local)
                    candidatos_livres.append(local_atual)

                    # Recalcula o estado completo só quando há melhoria
                    # (barato comparado a fazer isso para cada vizinho)
                    estado = calcular_estado(df, melhor_solucao, TEMPO_COBERTURA)

                    melhorou = True
                    break

            if melhorou:
                break

        if melhorou:
            rounds_sem_melhoria = 0
        else:
            rounds_sem_melhoria += 1

        parar_por_limite = (
            MAX_ROUNDS_SEM_MELHORIA is not None
            and rounds_sem_melhoria >= MAX_ROUNDS_SEM_MELHORIA
        )

        if parar_por_limite or not melhorou:
            break

    return melhor_solucao

# -----------------------------------------------------------------------
