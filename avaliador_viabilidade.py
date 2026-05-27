from busca_local import contar_ambulancias_por_tipo

def verificar_viabilidade(solucao, quantidade_maxima_por_tipo) -> bool:
    """
    Verifica se a solucao respeita todas as restricoes:
      - Exclusividade de regiao.
      - Disponibilidade por tipo: n_t(S) <= N_maximo_t para todo t.
    """
    regioes_utilizadas = [id_regiao for id_regiao, _ in solucao]
    if len(regioes_utilizadas) != len(set(regioes_utilizadas)):
        return False
    contagem = contar_ambulancias_por_tipo(solucao)
    return all(
        contagem.get(tipo, 0) <= limite
        for tipo, limite in quantidade_maxima_por_tipo.items()
    )