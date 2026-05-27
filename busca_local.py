"""
Busca local e funcoes de avaliacao de solucoes para o PMCS-FA.
"""


def calcular_funcao_objetivo(solucao, pontos_cobertos):
    lista_alocacoes = list(solucao)
    numero_alocacoes = len(lista_alocacoes)

    if numero_alocacoes == 0:
        return 0.0

    soma_cobertura_individual = sum(
        len(pontos_cobertos[id_regiao][tipo])
        for id_regiao, tipo in lista_alocacoes
    )

    soma_penalidade_sobreposicao = 0
    for indice_a in range(numero_alocacoes):
        id_regiao_a, tipo_a = lista_alocacoes[indice_a]
        for indice_b in range(numero_alocacoes):
            if indice_a == indice_b:
                continue
            id_regiao_b, tipo_b = lista_alocacoes[indice_b]
            soma_penalidade_sobreposicao += len(
                pontos_cobertos[id_regiao_a][tipo_a]
                & pontos_cobertos[id_regiao_b][tipo_b]
            )

    return float(soma_cobertura_individual - soma_penalidade_sobreposicao)


def obter_pontos_cobertos_pela_solucao(solucao, pontos_cobertos):
    """
    Retorna o conjunto de pontos de demanda cobertos por pelo menos
    uma alocacao da solucao.
    """
    cobertura_total = set()
    for id_regiao, tipo in solucao:
        cobertura_total |= pontos_cobertos[id_regiao][tipo]
    return cobertura_total


def contar_ambulancias_por_tipo(solucao):
    """Retorna um dicionario {tipo: quantidade_alocada} para a solucao."""
    contagem = {}
    for _, tipo in solucao:
        contagem[tipo] = contagem.get(tipo, 0) + 1
    return contagem


def busca_local(
    solucao_inicial,
    pontos_cobertos,
    regioes,
    tipos_ambulancia,
    quantidade_maxima_por_tipo,
    max_iteracoes_sem_melhora,
):
    """
    Aplica busca local a partir de uma solucao inicial, explorando tres
    vizinhancas em sequencia por passagem: troca, insercao e remocao.
    """
    melhor_solucao = set(solucao_inicial)
    melhor_fo = calcular_funcao_objetivo(melhor_solucao, pontos_cobertos)
    iteracoes_sem_melhora = 0

    limite_sem_melhora = (
        float("inf") if max_iteracoes_sem_melhora is None
        else max_iteracoes_sem_melhora
    )

    while iteracoes_sem_melhora < limite_sem_melhora:
        houve_melhora = False

        for id_regiao, tipo in list(melhor_solucao):
            solucao_sem_atual = melhor_solucao - {(id_regiao, tipo)}
            contagem_sem_atual = contar_ambulancias_por_tipo(solucao_sem_atual)
            regioes_sem_atual = {r for r, _ in solucao_sem_atual}

            for tipo_candidato in tipos_ambulancia:
                if (
                    contagem_sem_atual.get(tipo_candidato, 0)
                    >= quantidade_maxima_por_tipo[tipo_candidato]
                ):
                    continue
                for id_regiao_candidata in regioes:
                    if id_regiao_candidata in regioes_sem_atual:
                        continue
                    solucao_candidata = solucao_sem_atual | {
                        (id_regiao_candidata, tipo_candidato)
                    }
                    fo_candidata = calcular_funcao_objetivo(
                        solucao_candidata,
                        pontos_cobertos,
                    )
                    if fo_candidata > melhor_fo:
                        melhor_solucao = solucao_candidata
                        melhor_fo = fo_candidata
                        houve_melhora = True

        contagem_atual = contar_ambulancias_por_tipo(melhor_solucao)
        regioes_atuais = {r for r, _ in melhor_solucao}
        cobertura_atual = obter_pontos_cobertos_pela_solucao(
            melhor_solucao,
            pontos_cobertos,
        )

        for tipo in tipos_ambulancia:
            if contagem_atual.get(tipo, 0) >= quantidade_maxima_por_tipo[tipo]:
                continue
            for id_regiao in regioes:
                if contagem_atual.get(tipo, 0) >= quantidade_maxima_por_tipo[tipo]:
                    break
                if id_regiao in regioes_atuais:
                    continue
                if len(pontos_cobertos[id_regiao][tipo] - cobertura_atual) == 0:
                    continue
                solucao_candidata = melhor_solucao | {(id_regiao, tipo)}
                fo_candidata = calcular_funcao_objetivo(
                    solucao_candidata,
                    pontos_cobertos,
                )
                if fo_candidata > melhor_fo:
                    melhor_solucao = solucao_candidata
                    melhor_fo = fo_candidata
                    houve_melhora = True
                    contagem_atual = contar_ambulancias_por_tipo(melhor_solucao)
                    regioes_atuais = {r for r, _ in melhor_solucao}
                    cobertura_atual = obter_pontos_cobertos_pela_solucao(
                        melhor_solucao,
                        pontos_cobertos,
                    )

        for alocacao in list(melhor_solucao):
            solucao_candidata = melhor_solucao - {alocacao}
            fo_candidata = calcular_funcao_objetivo(
                solucao_candidata,
                pontos_cobertos,
            )
            if fo_candidata > melhor_fo:
                melhor_solucao = solucao_candidata
                melhor_fo = fo_candidata
                houve_melhora = True

        iteracoes_sem_melhora = 0 if houve_melhora else iteracoes_sem_melhora + 1

    return melhor_solucao, melhor_fo
