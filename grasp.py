"""
GRASP para o Problema de Máxima Cobertura com Sobreposição
para Frota de Ambulâncias (PMCS-FA)
"""

import numpy
import random
import time

from read_instance import ler_instancia, pre_computar_pontos_cobertos


# ============================================================
#  PARAMETROS — ajuste aqui conforme o problema real
# ============================================================

TIPOS_AMBULANCIA = {
    0: {"nome": "B",   "raio_cobertura_km": 3.0},
    1: {"nome": "A", "raio_cobertura_km": 5.0},
}

QUANTIDADE_MAXIMA_POR_TIPO = {0: 4, 1: 2}

PARAMETRO_ALPHA        = 0.3   
MAX_ITERACOES          = 100   # numero de iteracoes do GRASP
MAX_ITERACOES_SEM_MELHORA = None # criterio de parada da busca local
SEMENTE_ALEATORIA      = 42


# ============================================================
#  AVALIACAO DA SOLUCAO
# ============================================================
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


# ============================================================
# CONSTRUTIVO DO GRASP
# ============================================================
def construir_solucao(pontos_cobertos, regioes, tipos_ambulancia,
                      quantidade_maxima_por_tipo, parametro_alpha,
                      gerador_aleatorio):
    """
    Constrói uma solucao viavel.

    A cada passo do algoritmo construtivo:
      1. Calcula ganho(i,t|S) = |P_i^t menos C(S)| para cada candidato viavel.
         Um candidato e viavel se a regiao i nao esta ocupada e o numero de
         ambulancias do tipo t nao atingiu o limite maximo.
      2. Forma a Lista Restrita de Candidatos (RCL):
         candidatos com ganho >= ganho_maximo - alpha * (ganho_maximo - ganho_minimo).
      3. Sorteia uniformemente um elemento da RCL.
      4. Atualiza a solucao S, a cobertura C(S), as regioes ocupadas R(S)
         e o contador de ambulancias por tipo n_t(S).
    Para quando nao existe candidato viavel com ganho positivo.
    """
    solucao          = set()
    cobertura_atual  = set()
    regioes_ocupadas = set()
    contagem_tipo    = {tipo: 0 for tipo in tipos_ambulancia}

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
            break   # nenhuma alocacao adiciona ponto novo

        ganho_minimo = min(candidato[0] for candidato in candidatos_viaveis)
        limiar_rcl   = ganho_maximo - parametro_alpha * (ganho_maximo - ganho_minimo)
        lista_restrita_candidatos = [
            (id_regiao, tipo)
            for ganho, id_regiao, tipo in candidatos_viaveis
            if ganho >= limiar_rcl
        ]

        id_regiao_escolhida, tipo_escolhido = gerador_aleatorio.choice(
            lista_restrita_candidatos
        )

        solucao.add((id_regiao_escolhida, tipo_escolhido))
        cobertura_atual  |= pontos_cobertos[id_regiao_escolhida][tipo_escolhido]
        regioes_ocupadas.add(id_regiao_escolhida)
        contagem_tipo[tipo_escolhido] += 1

    return solucao


# ============================================================
# BUSCA LOCAL
# ============================================================

def busca_local(solucao_inicial, pontos_cobertos, regioes,
                tipos_ambulancia, quantidade_maxima_por_tipo,
                max_iteracoes_sem_melhora):
    """
    Aplica busca local a partir de uma solucao inicial, explorando tres
    vizinhancas em sequencia por passagem:

    TROCA    — substitui uma alocacao (i,t) presente na solucao por uma
               alocacao (i',t') viavel, aceitando a troca se melhorar a Função Objetivo.

    INSERÇÃO — acrescenta uma alocação (i,t) viavel com ganho marginal
               positivo, aceitando se melhorar a Funcao Objetivo. Apos cada insercao
               aceita, o estado de contagem e cobertura e recalculado
               para garantir viabilidade nas tentativas seguintes. O limite
               por tipo tambem e reavaliado dentro do laco de regioes para
               evitar que insercoes consecutivas de um mesmo tipo ultrapassem
               o maximo permitido.

    REMOCAO  — remove uma alocacao da solucao quando isso aumenta a Funcao Objetivo,
               o que ocorre quando a penalidade de sobreposicao superava
               a cobertura individual daquela alocacao.

    Criterio de parada: passagens consecutivas
    sem nenhuma melhoria na funcao objetivo.
    """
    melhor_solucao   = set(solucao_inicial)
    melhor_fo        = calcular_funcao_objetivo(melhor_solucao, pontos_cobertos)
    iteracoes_sem_melhora = 0

    limite_sem_melhora = (
        float("inf") if max_iteracoes_sem_melhora is None
        else max_iteracoes_sem_melhora
    )

    while iteracoes_sem_melhora < limite_sem_melhora:
        houve_melhora = False

        # ── Movimento TROCA ──────────────────────────────────
        for (id_regiao, tipo) in list(melhor_solucao):
            solucao_sem_atual    = melhor_solucao - {(id_regiao, tipo)}
            contagem_sem_atual   = contar_ambulancias_por_tipo(solucao_sem_atual)
            regioes_sem_atual    = {r for r, _ in solucao_sem_atual}

            for tipo_candidato in tipos_ambulancia:
                if contagem_sem_atual.get(tipo_candidato, 0) >= quantidade_maxima_por_tipo[tipo_candidato]:
                    continue
                for id_regiao_candidata in regioes:
                    if id_regiao_candidata in regioes_sem_atual:
                        continue
                    solucao_candidata = solucao_sem_atual | {(id_regiao_candidata, tipo_candidato)}
                    fo_candidata = calcular_funcao_objetivo(solucao_candidata, pontos_cobertos)
                    if fo_candidata > melhor_fo:
                        melhor_solucao = solucao_candidata
                        melhor_fo      = fo_candidata
                        houve_melhora  = True

        # ── INSERCAO ───────────────────────────────
        contagem_atual  = contar_ambulancias_por_tipo(melhor_solucao)
        regioes_atuais  = {r for r, _ in melhor_solucao}
        cobertura_atual = obter_pontos_cobertos_pela_solucao(melhor_solucao, pontos_cobertos)

        for tipo in tipos_ambulancia:
            if contagem_atual.get(tipo, 0) >= quantidade_maxima_por_tipo[tipo]:
                continue
            for id_regiao in regioes:
                # Reavalia o limite dentro do laco de regioes: se insercoes
                # anteriores ja preencheram a cota do tipo, interrompe.
                if contagem_atual.get(tipo, 0) >= quantidade_maxima_por_tipo[tipo]:
                    break
                if id_regiao in regioes_atuais:
                    continue
                if len(pontos_cobertos[id_regiao][tipo] - cobertura_atual) == 0:
                    continue   # ganho marginal zero, nao vale inserir
                solucao_candidata = melhor_solucao | {(id_regiao, tipo)}
                fo_candidata = calcular_funcao_objetivo(solucao_candidata, pontos_cobertos)
                if fo_candidata > melhor_fo:
                    melhor_solucao = solucao_candidata
                    melhor_fo      = fo_candidata
                    houve_melhora  = True
                    # Atualiza estado apos insercao aceita
                    contagem_atual  = contar_ambulancias_por_tipo(melhor_solucao)
                    regioes_atuais  = {r for r, _ in melhor_solucao}
                    cobertura_atual = obter_pontos_cobertos_pela_solucao(
                        melhor_solucao, pontos_cobertos
                    )

        # ── REMOCAO ────────────────────────────────
        for alocacao in list(melhor_solucao):
            solucao_candidata = melhor_solucao - {alocacao}
            fo_candidata = calcular_funcao_objetivo(solucao_candidata, pontos_cobertos)
            if fo_candidata > melhor_fo:
                melhor_solucao = solucao_candidata
                melhor_fo      = fo_candidata
                houve_melhora  = True

        iteracoes_sem_melhora = 0 if houve_melhora else iteracoes_sem_melhora + 1

    return melhor_solucao, melhor_fo


# ============================================================
#  GRASP
# ============================================================

def executar_grasp(dataframe, tipos_ambulancia, quantidade_maxima_por_tipo,
                   parametro_alpha, max_iteracoes, max_iteracoes_sem_melhora,
                   semente_aleatoria):
    
    gerador_aleatorio = random.Random(semente_aleatoria)
    regioes           = dataframe["local_id"].tolist()

    # ── Cabecalho ─────────────────────────────────────────────
    separador = "=" * 70
    print(separador)
    print("  GRASP — Problema de Maxima Cobertura com Sobreposicao (PMCS-FA)")
    print(separador)
    print(f"  Pontos de demanda        : {len(dataframe)}")
    print(f"  Regioes base candidatas  : {len(dataframe)}")
    for tipo, configuracao in tipos_ambulancia.items():
        print(
            f"  Tipo {tipo} ({configuracao['nome']:8s}): "
            f"raio = {configuracao['raio_cobertura_km']} km, "
            f"quantidade maxima = {quantidade_maxima_por_tipo[tipo]}"
        )
    print(f"  Parametro alpha (RCL)    : {parametro_alpha}  "
          f"(0 = guloso puro, 1 = aleatorio puro)")
    print(f"  Iteracoes GRASP          : {max_iteracoes}")
    print(f"  Parada busca local       : {max_iteracoes_sem_melhora} iteracoes sem melhora")
    print()

    # ── Pre-processamento ──────────────────────────────────────
    print("  Pre-computando pontos cobertos por regiao e tipo ...",
          end=" ", flush=True)
    tempo_inicio_precomputo = time.time()
    pontos_cobertos = pre_computar_pontos_cobertos(dataframe, tipos_ambulancia)
    tempo_precomputo = time.time() - tempo_inicio_precomputo
    print(f"concluido em {tempo_precomputo:.1f}s\n")

    melhor_solucao   = set()
    melhor_fo        = -1e18
    historico_fo     = []
    tempo_inicio_grasp = time.time()

    for numero_iteracao in range(1, max_iteracoes + 1):

        # construtivo GRASP
        solucao_construida = construir_solucao(
            pontos_cobertos, regioes, tipos_ambulancia,
            quantidade_maxima_por_tipo, parametro_alpha, gerador_aleatorio
        )

        # # busca local
        # solucao_apos_busca_local, fo_apos_busca_local = busca_local(
        #     solucao_construida, pontos_cobertos, regioes,
        #     tipos_ambulancia, quantidade_maxima_por_tipo,
        #     max_iteracoes_sem_melhora
        # )

        # historico_fo.append(fo_apos_busca_local)

        # Atualiza melhor solucao global
        # if fo_apos_busca_local > melhor_fo:
        #     melhor_fo      = fo_apos_busca_local
        #     melhor_solucao = set(solucao_apos_busca_local)
        #     percentual_cobertura = (
        #         100.0 * len(obter_pontos_cobertos_pela_solucao(melhor_solucao, pontos_cobertos))
        #         / len(dataframe)
        #     )
        #     print(
        #         f"  [Iteracao {numero_iteracao:3d}]  "
        #         f"Funcao objetivo = {melhor_fo:8.2f}  "
        #         f"Alocacoes = {len(melhor_solucao):2d}  "
        #         f"Cobertura = {percentual_cobertura:.1f}%"
        #     )
        melhor_solucao = solucao_construida
        print(
                f"  [Iteracao {numero_iteracao:3d}]  "
                # f"Funcao objetivo = {melhor_fo:8.2f}  "
                f"Alocacoes = {melhor_solucao}  "
            )

    tempo_total = time.time() - tempo_inicio_grasp

    # ── Verificacao de viabilidade ────────────────────────────
    assert verificar_viabilidade(melhor_solucao, quantidade_maxima_por_tipo), (
        f"ERRO: solucao final e inviavel! "
        f"Contagem por tipo: {contar_ambulancias_por_tipo(melhor_solucao)}"
    )

    pontos_cobertos_solucao = obter_pontos_cobertos_pela_solucao(
        melhor_solucao, pontos_cobertos
    )
    contagem_final = contar_ambulancias_por_tipo(melhor_solucao)
    array_historico = numpy.array(historico_fo)

    # ── Resultado final ───────────────────────────────────────
    print()
    print(separador)
    print("  RESULTADO FINAL")
    print(separador)
    print(f"  Funcao objetivo          : {melhor_fo:.2f}")
    print(
        f"  Pontos cobertos          : "
        f"{len(pontos_cobertos_solucao)} de {len(dataframe)}  "
        f"({100.0 * len(pontos_cobertos_solucao) / len(dataframe):.1f}%)"
    )
    print(f"  Total de alocacoes       : {len(melhor_solucao)}")
    print(f"  Tempo de execucao        : {tempo_total:.1f}s")
    print()

    # Tabela de alocacoes
    cabecalho_tabela = (
        f"  {'Regiao':>7}  {'Tipo':>4}  {'Nome':10}  "
        f"{'Latitude':>12}  {'Longitude':>13}  {'Pontos cobertos':>15}"
    )
    print(cabecalho_tabela)
    print("  " + "-" * (len(cabecalho_tabela) - 2))
    for id_regiao, tipo in sorted(melhor_solucao):
        linha = dataframe[dataframe["local_id"] == id_regiao].iloc[0]
        print(
            f"  {id_regiao:7d}  {tipo:4d}  "
            f"{tipos_ambulancia[tipo]['nome']:10}  "
            f"{linha['latitude']:12.5f}  {linha['longitude']:13.5f}  "
            f"{len(pontos_cobertos[id_regiao][tipo]):15d}"
        )

    print()
    print("  Ambulancias utilizadas por tipo:")
    for tipo, configuracao in tipos_ambulancia.items():
        utilizadas  = contagem_final.get(tipo, 0)
        disponivel  = quantidade_maxima_por_tipo[tipo]
        barra       = "█" * utilizadas + "░" * (disponivel - utilizadas)
        print(
            f"    Tipo {tipo} ({configuracao['nome']:8s}): "
            f"{utilizadas} de {disponivel}  [{barra}]"
        )

    # print()
    # print("  Estatisticas das iteracoes do GRASP:")
    # print(f"    Melhor funcao objetivo   : {array_historico.max():.2f}")
    # print(f"    Media da funcao objetivo : {array_historico.mean():.2f}")
    # print(f"    Pior funcao objetivo     : {array_historico.min():.2f}")
    # print(f"    Desvio padrao            : {array_historico.std():.2f}")
    print(separador)

    return melhor_solucao, melhor_fo, pontos_cobertos, historico_fo


if __name__ == "__main__":
    dataframe = ler_instancia("instancia.csv")

    melhor_solucao, melhor_fo, pontos_cobertos, historico_fo = executar_grasp(
        dataframe                  = dataframe,
        tipos_ambulancia           = TIPOS_AMBULANCIA,
        quantidade_maxima_por_tipo = QUANTIDADE_MAXIMA_POR_TIPO,
        parametro_alpha            = PARAMETRO_ALPHA,
        max_iteracoes              = MAX_ITERACOES,
        max_iteracoes_sem_melhora  = MAX_ITERACOES_SEM_MELHORA,
        semente_aleatoria          = SEMENTE_ALEATORIA,
    )
