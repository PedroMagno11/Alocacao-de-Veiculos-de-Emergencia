def construir_solucao_dm(
    pontos_cobertos,
    regioes,
    tipos_ambulancia,
    quantidade_maxima_por_tipo,
    parametro_alpha,
    gerador_aleatorio,
    pesos_candidatos=None,
):
    if pesos_candidatos is None:
        pesos_candidatos = {}

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

                ganho_original = len(
                    pontos_cobertos[id_regiao][tipo] - cobertura_atual
                )

                candidato = (id_regiao, tipo)
                if hasattr(pesos_candidatos, "peso_contextual"):
                    peso = pesos_candidatos.peso_contextual(candidato, solucao)
                else:
                    peso = pesos_candidatos.get(candidato, 1.0)

                ganho_dm = ganho_original * peso

                candidatos_viaveis.append(
                    (ganho_dm, id_regiao, tipo)
                )

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
