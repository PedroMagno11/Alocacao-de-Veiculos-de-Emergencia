import random
from pandas import DataFrame


# -----------------------------------------------------------------------
# Caches em escopo de módulo — populados uma única vez por processo.
# Eliminam df.loc[] dentro de qualquer loop em todas as funções.
# -----------------------------------------------------------------------

_demanda_complexa:     dict = {}
_demanda_simples:      dict = {}
_cobertura_cache:      dict = {}   # (local_id, tempo_cobertura) → set de regiões
_indice_reverso_cache: dict = {}   # tempo_cobertura → {regiao: set de locais}
_locais_cache:         dict = {}   # tempo_cobertura → list de local_ids


def _garantir_cache(df: DataFrame, TEMPO_COBERTURA: int):
    global _demanda_complexa, _demanda_simples, _cobertura_cache
    global _indice_reverso_cache, _locais_cache

    locais = list(df.index.astype(str))
    _locais_cache[TEMPO_COBERTURA] = locais

    if not _demanda_complexa:
        _demanda_complexa = df["demanda_complexa"].to_dict()
        _demanda_simples  = df["demanda_simples"].to_dict()

    if (locais[0], TEMPO_COBERTURA) not in _cobertura_cache:
        for local_id in locais:
            _cobertura_cache[(local_id, TEMPO_COBERTURA)] = set(
                df.loc[local_id, f"cobertura_{TEMPO_COBERTURA}_min"]
            )

    if TEMPO_COBERTURA not in _indice_reverso_cache:
        regiao_para_locais: dict = {}
        for local_id in locais:
            for r in _cobertura(local_id, TEMPO_COBERTURA):
                if r not in regiao_para_locais:
                    regiao_para_locais[r] = set()
                regiao_para_locais[r].add(local_id)
        _indice_reverso_cache[TEMPO_COBERTURA] = regiao_para_locais


def _somar(regioes: set, tipo: str) -> float:
    if not regioes:
        return 0.0
    if tipo == "A":
        return sum(_demanda_complexa[r] for r in regioes)
    return sum(_demanda_simples[r] for r in regioes)


def _cobertura(local_id: str, TEMPO_COBERTURA: int) -> set:
    return _cobertura_cache[(local_id, TEMPO_COBERTURA)]


# -----------------------------------------------------------------------
# Score completo
# -----------------------------------------------------------------------

def calcular_score(df: DataFrame, solucao, TEMPO_COBERTURA):
    _garantir_cache(df, TEMPO_COBERTURA)

    regioes_cobertas_total  = set()
    score_total             = 0.0
    score_ambulancia_tipo_A = 0.0
    score_ambulancia_tipo_B = 0.0

    for local_id, tipo_ambulancia in solucao:
        regioes_cobertas = _cobertura(local_id, TEMPO_COBERTURA)
        novas_regioes    = regioes_cobertas - regioes_cobertas_total

        if not novas_regioes:
            continue

        ganho = _somar(novas_regioes, tipo_ambulancia)

        if tipo_ambulancia == "A":
            score_ambulancia_tipo_A += ganho
        elif tipo_ambulancia == "B":
            score_ambulancia_tipo_B += ganho

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
    _garantir_cache(df, TEMPO_COBERTURA)

    regioes_cobertas_total = set()
    cobertura_por_regiao = {}
    contribuicao = []

    score_total = 0.0
    score_ambulancia_tipo_A = 0.0
    score_ambulancia_tipo_B = 0.0

    for local_id, tipo_ambulancia in solucao:
        regioes_cobertas = _cobertura(local_id, TEMPO_COBERTURA)
        novas_regioes    = regioes_cobertas - regioes_cobertas_total

        for r in regioes_cobertas:
            cobertura_por_regiao[r] = cobertura_por_regiao.get(r, 0) + 1

        if novas_regioes:
            ganho = _somar(novas_regioes, tipo_ambulancia)

            if tipo_ambulancia == "A":
                score_ambulancia_tipo_A += ganho
            elif tipo_ambulancia == "B":
                score_ambulancia_tipo_B += ganho

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
# Delta incremental
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
    _garantir_cache(df, TEMPO_COBERTURA)

    cobertura_por_regiao   = estado["cobertura_por_regiao"]
    regioes_cobertas_total = estado["regioes_cobertas_total"]

    regioes_antigo = _cobertura(local_antigo, TEMPO_COBERTURA)
    regioes_novo   = _cobertura(novo_local,   TEMPO_COBERTURA)

    # Regiões que ficam descobertas ao remover local_antigo:
    # apenas as que ele cobre sozinho (contagem == 1)
    regioes_perdidas = {
        r for r in regioes_antigo
        if cobertura_por_regiao.get(r, 0) == 1
    }
    perda = _somar(regioes_perdidas, tipo_ambulancia)

    # Regiões ainda cobertas após remover local_antigo
    # (inclui as que outro local também cobre)
    regioes_cobertas_sem_antigo = regioes_cobertas_total - regioes_perdidas

    # Regiões efetivamente novas que novo_local traz
    regioes_ganhas = regioes_novo - regioes_cobertas_sem_antigo
    ganho = _somar(regioes_ganhas, tipo_ambulancia)

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
    _garantir_cache(df, TEMPO_DE_COBERTURA)
    regioes_cobertas = _cobertura(local_id, TEMPO_DE_COBERTURA)
    novas_regioes    = regioes_cobertas - regioes_ja_cobertas
    return _somar(novas_regioes, tipo_ambulancia)


# -----------------------------------------------------------------------

def escolher_da_lista_de_candidatos_restritos(avaliacoes, ALPHA):
    avaliacoes   = sorted(avaliacoes, key=lambda x: x["ganho"], reverse=True)
    melhor_ganho = avaliacoes[0]["ganho"]
    pior_ganho   = avaliacoes[-1]["ganho"]
    limite       = melhor_ganho - ALPHA * (melhor_ganho - pior_ganho)
    lista_restrita = [a for a in avaliacoes if a["ganho"] >= limite]
    return random.choice(lista_restrita)


# -----------------------------------------------------------------------
# Construção de solução — incremental + índice reverso + cache de demanda
# -----------------------------------------------------------------------

def construir_solucao(df: DataFrame, QTD_AMBULANCIAS_A, QTD_AMBULANCIAS_B, ALPHA, TEMPO_COBERTURA):
    _garantir_cache(df, TEMPO_COBERTURA)

    solucao = []
    regioes_ja_cobertas = set()

    ambulancias_para_alocar = ["A"] * QTD_AMBULANCIAS_A + ["B"] * QTD_AMBULANCIAS_B
    random.shuffle(ambulancias_para_alocar)

    locais_disponiveis     = list(_locais_cache[TEMPO_COBERTURA])
    locais_disponiveis_set = set(locais_disponiveis)

    # Índice reverso já pré-computado no cache global
    regiao_para_locais = _indice_reverso_cache[TEMPO_COBERTURA]

    # Ganhos iniciais (1ª ambulância, regioes_ja_cobertas ainda vazio)
    tipo_inicial = ambulancias_para_alocar[0]
    ganhos = {
        local_id: _somar(_cobertura(local_id, TEMPO_COBERTURA), tipo_inicial)
        for local_id in locais_disponiveis
    }

    for idx_amb, tipo_ambulancia in enumerate(ambulancias_para_alocar):

        # Mudança de tipo A <-> B: recalcula todos (ocorre no máximo uma vez)
        if idx_amb > 0 and tipo_ambulancia != ambulancias_para_alocar[idx_amb - 1]:
            ganhos = {
                local_id: _somar(
                    _cobertura(local_id, TEMPO_COBERTURA) - regioes_ja_cobertas,
                    tipo_ambulancia
                )
                for local_id in locais_disponiveis
            }

        avaliacoes = [
            {"local_id": lid, "tipo_ambulancia": tipo_ambulancia, "ganho": ganhos[lid]}
            for lid in locais_disponiveis
        ]

        escolhido       = escolher_da_lista_de_candidatos_restritos(avaliacoes, ALPHA)
        local_escolhido = escolhido["local_id"]
        tipo_escolhido  = escolhido["tipo_ambulancia"]

        solucao.append((local_escolhido, tipo_escolhido))

        regioes_novas_cobertas = _cobertura(local_escolhido, TEMPO_COBERTURA) - regioes_ja_cobertas
        regioes_ja_cobertas.update(regioes_novas_cobertas)
        locais_disponiveis.remove(local_escolhido)
        locais_disponiveis_set.remove(local_escolhido)
        del ganhos[local_escolhido]

        # Índice reverso -> só recalcula locais realmente afetados
        locais_afetados = set()
        for r in regioes_novas_cobertas:
            locais_afetados.update(regiao_para_locais.get(r, set()))
        locais_afetados &= locais_disponiveis_set

        for local_id in locais_afetados:
            ganhos[local_id] = _somar(
                _cobertura(local_id, TEMPO_COBERTURA) - regioes_ja_cobertas,
                tipo_ambulancia
            )

    return solucao


# -----------------------------------------------------------------------
# Busca local com avaliação incremental (delta)
# -----------------------------------------------------------------------

def busca_local(
    df: DataFrame,
    solucao,
    TEMPO_COBERTURA,
    MAX_ROUNDS_SEM_MELHORIA: int = None,
    MAX_CANDIDATOS_POR_ROUND: int = None,
):
    """
    Busca local com avaliação incremental por delta.

    Parâmetros
    ----------
    MAX_ROUNDS_SEM_MELHORIA : int | None
        Para após N rounds consecutivos sem melhoria.
        None = convergência total (só para quando nenhum vizinho melhora).

    MAX_CANDIDATOS_POR_ROUND : int | None
        Limita quantos candidatos livres são avaliados por posição por round.
        None = avalia todos (~3011 com 3016 locais).
        Valores entre 100–300 reduzem o tempo por round de forma linear
        sem sacrificar muito a qualidade — a diversidade entre iterações
        do GRASP compensa a cobertura parcial da vizinhança.

        Exemplo de impacto:
          Sem limite : ~15.000 deltas/round  (5 pos × 3011 cand)
          Limite=200 : ~1.000 deltas/round   (5 pos × 200 cand)  → 15x mais rápido/round
    """
    _garantir_cache(df, TEMPO_COBERTURA)

    melhor_solucao = solucao.copy()
    estado         = calcular_estado(df, melhor_solucao, TEMPO_COBERTURA)

    locais_usados     = {local_id for local_id, _ in melhor_solucao}
    candidatos_livres = [lid for lid in _locais_cache[TEMPO_COBERTURA] if lid not in locais_usados]

    rounds_sem_melhoria = 0

    while True:
        melhorou = False

        # Amostra aleatória de candidatos para este round, se limite definido.
        # Embaralhar a cada round garante que candidatos diferentes são testados
        # ao longo dos rounds, mantendo cobertura ampla da vizinhança.
        if MAX_CANDIDATOS_POR_ROUND is not None and MAX_CANDIDATOS_POR_ROUND < len(candidatos_livres):
            candidatos_do_round = random.sample(candidatos_livres, MAX_CANDIDATOS_POR_ROUND)
        else:
            candidatos_do_round = candidatos_livres

        for indice, (local_atual, tipo_ambulancia) in enumerate(melhor_solucao):
            for novo_local in candidatos_do_round:

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
                    melhor_solucao[indice] = (novo_local, tipo_ambulancia)
                    candidatos_livres.remove(novo_local)
                    candidatos_livres.append(local_atual)
                    estado   = calcular_estado(df, melhor_solucao, TEMPO_COBERTURA)
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

        if parar_por_limite:
            break

        # Sem limite definido: para quando nenhum vizinho melhora (convergência)
        if MAX_ROUNDS_SEM_MELHORIA is None and not melhorou:
            break

    return melhor_solucao
