
from collections import Counter


def minerar_frequencia(memoria_elite, frequencia_minima=0.3):
    solucoes = memoria_elite.obter_solucoes()

    if not solucoes:
        return {}

    contador = Counter()

    for item in solucoes:
        for candidato in item["solucao"]:
            contador[candidato] += 1

    quantidade_solucoes = len(solucoes)
    pesos = {}

    for candidato, frequencia in contador.items():
        percentual = frequencia / quantidade_solucoes

        if percentual >= frequencia_minima:
            pesos[candidato] = 1.0 + percentual

    return pesos
