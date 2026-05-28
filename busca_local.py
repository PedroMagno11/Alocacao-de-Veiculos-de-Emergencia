"""
Busca local e funcoes de avaliacao de solucoes para o PMCS-FA.

Esta versao respeita a formulacao do PDF:

    max sum |P_i^t| x_i^t
        - sum |P_i^t inter P_j^t'| y_ij^tt', com j != i

Ou seja, a funcao objetivo NAO e tratada apenas como uniao dos pontos
cobertos. A sobreposicao entre duas alocacoes ativas e penalizada.

Otimizacoes implementadas:
  1. Pre-computacao das cardinalidades |P_i^t| e |P_i^t inter P_j^t'|.
  2. Avaliacao incremental por delta para insercao, remocao e troca.
  3. Estrategia first improvement: ao encontrar a primeira melhora, aceita
     imediatamente e reinicia a exploracao da vizinhanca.
"""

from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

Alocacao = Tuple[int, str]
PontosCobertos = Dict[int, Dict[str, FrozenSet[int]]]


@dataclass(frozen=True)
class PreCalculoCobertura:
    """
    Estrutura em memoria com areas cobertas e intersecoes pre-computadas.

    area[(i, t)] = |P_i^t|
    intersecao[((i, t), (j, t'))] = |P_i^t inter P_j^t'|

    A intersecao e guardada para pares ordenados, pois a formulacao do PDF
    usa somatorios em t, t', i e j, com j != i. Mesmo que a cardinalidade seja
    simetrica, guardar o par ordenado evita ambiguidades na avaliacao.
    """

    area: Dict[Alocacao, int]
    intersecao: Dict[Tuple[Alocacao, Alocacao], int]

    def obter_area(self, alocacao: Alocacao) -> int:
        return self.area.get(alocacao, 0)

    def obter_intersecao(self, primeira: Alocacao, segunda: Alocacao) -> int:
        if primeira[0] == segunda[0]:
            # A restricao do problema impede mais de uma ambulancia na mesma regiao
            # e a FO do PDF considera j != i na penalizacao.
            return 0
        return self.intersecao.get((primeira, segunda), 0)


def pre_computar_areas_e_intersecoes(
    pontos_cobertos: PontosCobertos,
    regioes: Iterable[int],
    tipos_ambulancia: Iterable[str],
) -> PreCalculoCobertura:
    """
    Pre-computa, em memoria, as cardinalidades de cobertura individual e as
    intersecoes entre todos os candidatos (regiao, tipo).

    Complexidade de pre-processamento: O(|R|^2 * |T|^2 * custo_intersecao).
    A vantagem e que a busca local passa a consultar inteiros em dicionario,
    em vez de recalcular intersecoes de conjuntos repetidamente.
    """
    candidatos: List[Alocacao] = [
        (int(id_regiao), tipo)
        for id_regiao in regioes
        for tipo in tipos_ambulancia
    ]

    area: Dict[Alocacao, int] = {}
    intersecao: Dict[Tuple[Alocacao, Alocacao], int] = {}

    for alocacao in candidatos:
        id_regiao, tipo = alocacao
        area[alocacao] = len(pontos_cobertos[id_regiao][tipo])

    for primeira in candidatos:
        pontos_primeira = pontos_cobertos[primeira[0]][primeira[1]]
        for segunda in candidatos:
            if primeira == segunda or primeira[0] == segunda[0]:
                continue
            pontos_segunda = pontos_cobertos[segunda[0]][segunda[1]]
            intersecao[(primeira, segunda)] = len(pontos_primeira & pontos_segunda)

    return PreCalculoCobertura(area=area, intersecao=intersecao)


def obter_pontos_cobertos_pela_solucao(
    solucao: Iterable[Alocacao],
    pontos_cobertos: PontosCobertos,
) -> Set[int]:
    """
    Retorna a uniao fisica dos pontos cobertos por pelo menos uma alocacao.

    Esta funcao e util para relatorios, mas nao deve ser confundida com a
    funcao objetivo do PMCS-FA quando ha penalizacao por sobreposicao.
    """
    cobertura_total: Set[int] = set()
    for id_regiao, tipo in solucao:
        cobertura_total |= set(pontos_cobertos[id_regiao][tipo])
    return cobertura_total


def contar_ambulancias_por_tipo(solucao: Iterable[Alocacao]) -> Dict[str, int]:
    """Retorna um dicionario {tipo: quantidade_alocada} para a solucao."""
    contagem: Dict[str, int] = {}
    for _, tipo in solucao:
        contagem[tipo] = contagem.get(tipo, 0) + 1
    return contagem


def calcular_funcao_objetivo(
    solucao: Iterable[Alocacao],
    pontos_cobertos: Optional[PontosCobertos] = None,
    pre_calculo: Optional[PreCalculoCobertura] = None,
) -> float:
    """
    Calcula a funcao objetivo do PMCS-FA conforme o PDF:

      soma das coberturas individuais - penalizacao das intersecoes ordenadas.

    Se pre_calculo nao for informado, ele e construido automaticamente usando
    pontos_cobertos e os elementos presentes na solucao. Para desempenho, passe
    sempre pre_calculo na busca local.
    """
    solucao_lista = list(solucao)
    if not solucao_lista:
        return 0.0

    if pre_calculo is None:
        if pontos_cobertos is None:
            raise ValueError("Informe pontos_cobertos ou pre_calculo.")
        regioes = sorted({id_regiao for id_regiao, _ in solucao_lista})
        tipos = sorted({tipo for _, tipo in solucao_lista})
        pre_calculo = pre_computar_areas_e_intersecoes(
            pontos_cobertos,
            regioes,
            tipos,
        )

    cobertura_individual = sum(
        pre_calculo.obter_area(alocacao)
        for alocacao in solucao_lista
    )

    penalizacao_sobreposicao = 0
    for primeira in solucao_lista:
        for segunda in solucao_lista:
            if primeira == segunda:
                continue
            penalizacao_sobreposicao += pre_calculo.obter_intersecao(
                primeira,
                segunda,
            )

    return float(cobertura_individual - penalizacao_sobreposicao)


def _delta_insercao(
    candidato: Alocacao,
    solucao_atual: Set[Alocacao],
    pre_calculo: PreCalculoCobertura,
) -> int:
    """Ganho de FO ao inserir candidato em S."""
    penalizacao_nova = 0
    for alocacao in solucao_atual:
        penalizacao_nova += pre_calculo.obter_intersecao(candidato, alocacao)
        penalizacao_nova += pre_calculo.obter_intersecao(alocacao, candidato)
    return pre_calculo.obter_area(candidato) - penalizacao_nova


def _delta_remocao(
    alocacao_removida: Alocacao,
    solucao_atual: Set[Alocacao],
    pre_calculo: PreCalculoCobertura,
) -> int:
    """Ganho de FO ao remover uma alocacao de S."""
    penalizacao_removida = 0
    for alocacao in solucao_atual:
        if alocacao == alocacao_removida:
            continue
        penalizacao_removida += pre_calculo.obter_intersecao(
            alocacao_removida,
            alocacao,
        )
        penalizacao_removida += pre_calculo.obter_intersecao(
            alocacao,
            alocacao_removida,
        )
    return -pre_calculo.obter_area(alocacao_removida) + penalizacao_removida


def _eh_viavel_para_insercao(
    candidato: Alocacao,
    regioes_ocupadas: Set[int],
    contagem_tipo: Dict[str, int],
    quantidade_maxima_por_tipo: Dict[str, int],
) -> bool:
    id_regiao, tipo = candidato
    if id_regiao in regioes_ocupadas:
        return False
    return contagem_tipo.get(tipo, 0) < quantidade_maxima_por_tipo[tipo]


def _tentar_primeira_melhora_troca(
    solucao_atual: Set[Alocacao],
    regioes: Iterable[int],
    tipos_ambulancia: Iterable[str],
    quantidade_maxima_por_tipo: Dict[str, int],
    pre_calculo: PreCalculoCobertura,
) -> Optional[Tuple[Set[Alocacao], int]]:
    """Retorna a primeira solucao melhor por troca, se existir."""
    for alocacao_removida in list(solucao_atual):
        solucao_sem = set(solucao_atual)
        solucao_sem.remove(alocacao_removida)
        regioes_ocupadas = {id_regiao for id_regiao, _ in solucao_sem}
        contagem_tipo = contar_ambulancias_por_tipo(solucao_sem)
        delta_remocao = _delta_remocao(alocacao_removida, solucao_atual, pre_calculo)

        for tipo_candidato in tipos_ambulancia:
            if contagem_tipo.get(tipo_candidato, 0) >= quantidade_maxima_por_tipo[tipo_candidato]:
                continue
            for id_regiao_candidata in regioes:
                candidato = (int(id_regiao_candidata), tipo_candidato)
                if candidato == alocacao_removida:
                    continue
                if not _eh_viavel_para_insercao(
                    candidato,
                    regioes_ocupadas,
                    contagem_tipo,
                    quantidade_maxima_por_tipo,
                ):
                    continue

                delta_total = delta_remocao + _delta_insercao(
                    candidato,
                    solucao_sem,
                    pre_calculo,
                )
                if delta_total > 0:
                    nova_solucao = set(solucao_sem)
                    nova_solucao.add(candidato)
                    return nova_solucao, delta_total
    return None


def _tentar_primeira_melhora_insercao(
    solucao_atual: Set[Alocacao],
    regioes: Iterable[int],
    tipos_ambulancia: Iterable[str],
    quantidade_maxima_por_tipo: Dict[str, int],
    pre_calculo: PreCalculoCobertura,
) -> Optional[Tuple[Set[Alocacao], int]]:
    """Retorna a primeira solucao melhor por insercao, se existir."""
    regioes_ocupadas = {id_regiao for id_regiao, _ in solucao_atual}
    contagem_tipo = contar_ambulancias_por_tipo(solucao_atual)

    for tipo in tipos_ambulancia:
        if contagem_tipo.get(tipo, 0) >= quantidade_maxima_por_tipo[tipo]:
            continue
        for id_regiao in regioes:
            candidato = (int(id_regiao), tipo)
            if not _eh_viavel_para_insercao(
                candidato,
                regioes_ocupadas,
                contagem_tipo,
                quantidade_maxima_por_tipo,
            ):
                continue
            delta = _delta_insercao(candidato, solucao_atual, pre_calculo)
            if delta > 0:
                nova_solucao = set(solucao_atual)
                nova_solucao.add(candidato)
                return nova_solucao, delta
    return None


def _tentar_primeira_melhora_remocao(
    solucao_atual: Set[Alocacao],
    pre_calculo: PreCalculoCobertura,
) -> Optional[Tuple[Set[Alocacao], int]]:
    """Retorna a primeira solucao melhor por remocao, se existir."""
    for alocacao in list(solucao_atual):
        delta = _delta_remocao(alocacao, solucao_atual, pre_calculo)
        if delta > 0:
            nova_solucao = set(solucao_atual)
            nova_solucao.remove(alocacao)
            return nova_solucao, delta
    return None


def busca_local(
    solucao_inicial,
    pontos_cobertos,
    regioes,
    tipos_ambulancia,
    quantidade_maxima_por_tipo,
    max_iteracoes_sem_melhora=None,
    pre_calculo: Optional[PreCalculoCobertura] = None,
):
    """
    Aplica busca local com first improvement.

    Ordem de vizinhancas por ciclo:
      1. troca: remove uma alocacao e adiciona outra;
      2. insercao: adiciona uma alocacao viavel;
      3. remocao: remove uma alocacao que esteja prejudicando a FO.

    Quando uma melhora e encontrada, ela e aceita imediatamente e a busca
    reinicia a partir da primeira vizinhanca. Isso evita varrer toda a
    vizinhanca quando uma melhoria ja foi localizada.
    """
    if pre_calculo is None:
        pre_calculo = pre_computar_areas_e_intersecoes(
            pontos_cobertos,
            regioes,
            tipos_ambulancia,
        )

    melhor_solucao: Set[Alocacao] = set(solucao_inicial)
    melhor_fo = calcular_funcao_objetivo(
        melhor_solucao,
        pre_calculo=pre_calculo,
    )

    iteracoes_sem_melhora = 0
    limite_sem_melhora = (
        float("inf") if max_iteracoes_sem_melhora is None
        else max_iteracoes_sem_melhora
    )

    while iteracoes_sem_melhora < limite_sem_melhora:
        movimento = (
            _tentar_primeira_melhora_troca(
                melhor_solucao,
                regioes,
                tipos_ambulancia,
                quantidade_maxima_por_tipo,
                pre_calculo,
            )
            or _tentar_primeira_melhora_insercao(
                melhor_solucao,
                regioes,
                tipos_ambulancia,
                quantidade_maxima_por_tipo,
                pre_calculo,
            )
            or _tentar_primeira_melhora_remocao(
                melhor_solucao,
                pre_calculo,
            )
        )

        if movimento is None:
            iteracoes_sem_melhora += 1
            continue

        melhor_solucao, delta = movimento
        melhor_fo += delta
        iteracoes_sem_melhora = 0

    return melhor_solucao, float(melhor_fo)
