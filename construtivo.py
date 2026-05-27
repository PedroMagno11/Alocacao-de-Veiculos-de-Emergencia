"""
Construtivo do GRASP para o PMCS-FA.
"""


def construir_solucao(
    pontos_cobertos,
    regioes,
    tipos_ambulancia,
    quantidade_maxima_por_tipo,
    parametro_alpha,
    gerador_aleatorio,
):
    """
    Constroi uma solucao viavel.

    A cada passo do algoritmo construtivo:
      1. Calcula ganho(i,t|S) = |P_i^t menos C(S)| para cada candidato viavel.
      2. Forma a Lista Restrita de Candidatos (RCL).
      3. Sorteia uniformemente um elemento da RCL.
      4. Atualiza solucao, cobertura, regioes ocupadas e contador por tipo.
    Para quando nao existe candidato viavel com ganho positivo.
    """
    solucao = set()
    cobertura_atual = set()
    regioes_ocupadas = set()
    contagem_tipo = {tipo: 0 for tipo in tipos_ambulancia}

    while True:
        candidatos_viaveis = []
        for tipo in tipos_ambulancia:
            if contagem_tipo[tipo] >= quantidade_maxima_por_tipo[tipo]:
                continue
            for id_regiao in regioes:
                if id_regiao in regioes_ocupadas:
                    continue
                ganho = len(pontos_cobertos[id_regiao][tipo] - cobertura_atual)
                candidatos_viaveis.append((ganho, id_regiao, tipo))

        if not candidatos_viaveis:
            break

        ganho_maximo = max(candidato[0] for candidato in candidatos_viaveis)
        if ganho_maximo == 0:
            break

        ganho_minimo = min(candidato[0] for candidato in candidatos_viaveis)
        limiar_rcl = ganho_maximo - parametro_alpha * (
            ganho_maximo - ganho_minimo
        )
        lista_restrita_candidatos = [
            (id_regiao, tipo)
            for ganho, id_regiao, tipo in candidatos_viaveis
            if ganho >= limiar_rcl
        ]

        id_regiao_escolhida, tipo_escolhido = gerador_aleatorio.choice(
            lista_restrita_candidatos
        )

        solucao.add((id_regiao_escolhida, tipo_escolhido))
        cobertura_atual |= pontos_cobertos[id_regiao_escolhida][tipo_escolhido]
        regioes_ocupadas.add(id_regiao_escolhida)
        contagem_tipo[tipo_escolhido] += 1

    return solucao
