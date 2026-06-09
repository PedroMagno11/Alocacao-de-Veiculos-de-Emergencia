"""
Mineracao de padroes para o DM-GRASP.

A memoria elite e tratada como uma base transacional:
  - cada solucao elite e uma transacao;
  - cada alocacao (regiao, tipo) e um item.

O minerador abaixo segue a ideia do FPMax*: encontrar conjuntos frequentes
maximos, isto e, padroes frequentes que nao estao contidos em nenhum outro
padrao frequente maior. A implementacao usa uma representacao vertical por
tidsets para manter o codigo simples e adequado ao tamanho das solucoes do
PMCS-FA, sem depender de bibliotecas externas.
"""

from dataclasses import dataclass
from math import ceil
from typing import Dict, FrozenSet, Iterable, List, Sequence, Set, Tuple


Alocacao = Tuple[int, int]


@dataclass(frozen=True)
class PadraoMaximo:
    itens: FrozenSet[Alocacao]
    suporte: int
    percentual: float


@dataclass
class ResultadoFPMaxEstrela:
    pesos_itens: Dict[Alocacao, float]
    padroes_maximos: List[PadraoMaximo]
    suporte_minimo: int
    quantidade_transacoes: int

    def get(self, candidato: Alocacao, padrao: float = 1.0) -> float:
        """Mantem compatibilidade com o uso antigo como dicionario de pesos."""
        return self.pesos_itens.get(candidato, padrao)

    def __len__(self) -> int:
        return len(self.padroes_maximos)

    @property
    def quantidade_itens_ponderados(self) -> int:
        return len(self.pesos_itens)

    def peso_contextual(self, candidato: Alocacao, solucao_parcial: Iterable[Alocacao]) -> float:
        """
        Calcula o peso do candidato considerando os padroes maximos.

        Alem do peso individual, o candidato recebe bonus quando completa um
        padrao que ja esta parcialmente presente na solucao construida. Isso e
        o ponto em que a mineracao deixa de ser apenas frequencia de itens
        isolados e passa a influenciar combinacoes de alocacoes.
        """
        peso = self.get(candidato, 1.0)
        solucao_parcial = set(solucao_parcial)

        if not solucao_parcial:
            return peso

        maior_bonus = 0.0
        for padrao in self.padroes_maximos:
            if candidato not in padrao.itens or len(padrao.itens) <= 1:
                continue

            itens_presentes = len(padrao.itens & solucao_parcial)
            if itens_presentes == 0:
                continue

            afinidade = itens_presentes / (len(padrao.itens) - 1)
            bonus = padrao.percentual * afinidade
            if bonus > maior_bonus:
                maior_bonus = bonus

        return peso + maior_bonus


def _calcular_suporte_minimo(frequencia_minima: float, quantidade_transacoes: int) -> int:
    if quantidade_transacoes <= 0:
        return 1
    if frequencia_minima <= 1:
        return max(1, ceil(frequencia_minima * quantidade_transacoes))
    return max(1, int(frequencia_minima))


def _obter_transacoes(memoria_elite) -> List[FrozenSet[Alocacao]]:
    transacoes = []
    for item in memoria_elite.obter_solucoes():
        transacao = frozenset(item["solucao"])
        if transacao:
            transacoes.append(transacao)
    return transacoes


def _ordenar_itens_por_suporte(
    item_tidsets: Dict[Alocacao, FrozenSet[int]]
) -> List[Tuple[Alocacao, FrozenSet[int]]]:
    return sorted(
        item_tidsets.items(),
        key=lambda par: (-len(par[1]), par[0]),
    )


def _filtrar_maximos(
    candidatos: Sequence[Tuple[FrozenSet[Alocacao], int]],
    quantidade_transacoes: int,
) -> List[PadraoMaximo]:
    ordenados = sorted(
        candidatos,
        key=lambda par: (-len(par[0]), -par[1], sorted(par[0])),
    )

    maximos: List[PadraoMaximo] = []
    conjuntos_maximos: List[FrozenSet[Alocacao]] = []

    for itens, suporte in ordenados:
        if any(itens < conjunto_maior for conjunto_maior in conjuntos_maximos):
            continue

        conjuntos_maximos.append(itens)
        maximos.append(
            PadraoMaximo(
                itens=itens,
                suporte=suporte,
                percentual=suporte / quantidade_transacoes,
            )
        )

    return maximos


def _minerar_conjuntos_frequentes_maximos(
    transacoes: Sequence[FrozenSet[Alocacao]],
    suporte_minimo: int,
) -> List[PadraoMaximo]:
    item_tidsets: Dict[Alocacao, Set[int]] = {}

    for indice_transacao, transacao in enumerate(transacoes):
        for item in transacao:
            item_tidsets.setdefault(item, set()).add(indice_transacao)

    itens_frequentes = {
        item: frozenset(tids)
        for item, tids in item_tidsets.items()
        if len(tids) >= suporte_minimo
    }

    candidatos_maximos: List[Tuple[FrozenSet[Alocacao], int]] = []

    def expandir(
        prefixo: Tuple[Alocacao, ...],
        tidset_prefixo: FrozenSet[int],
        extensoes: Sequence[Tuple[Alocacao, FrozenSet[int]]],
    ) -> None:
        encontrou_extensao = False

        for indice, (item, tidset_item) in enumerate(extensoes):
            novo_tidset = tidset_item if not prefixo else tidset_prefixo & tidset_item
            suporte = len(novo_tidset)
            if suporte < suporte_minimo:
                continue

            novo_prefixo = prefixo + (item,)
            novas_extensoes = []

            for proximo_item, proximo_tidset in extensoes[indice + 1:]:
                intersecao = novo_tidset & proximo_tidset
                if len(intersecao) >= suporte_minimo:
                    novas_extensoes.append((proximo_item, intersecao))

            encontrou_extensao = True
            if novas_extensoes:
                expandir(novo_prefixo, novo_tidset, novas_extensoes)
            else:
                candidatos_maximos.append((frozenset(novo_prefixo), suporte))

        if prefixo and not encontrou_extensao:
            candidatos_maximos.append((frozenset(prefixo), len(tidset_prefixo)))

    expandir(tuple(), frozenset(range(len(transacoes))), _ordenar_itens_por_suporte(itens_frequentes))
    return _filtrar_maximos(candidatos_maximos, len(transacoes))


def _calcular_pesos_itens(padroes_maximos: Sequence[PadraoMaximo]) -> Dict[Alocacao, float]:
    pesos: Dict[Alocacao, float] = {}

    for padrao in padroes_maximos:
        bonus_tamanho = 1.0 + (len(padrao.itens) - 1) / max(len(padrao.itens), 1)
        peso_padrao = 1.0 + padrao.percentual * bonus_tamanho

        for item in padrao.itens:
            pesos[item] = max(pesos.get(item, 1.0), peso_padrao)

    return pesos


def minerar_fpmax_estrela(memoria_elite, frequencia_minima=0.3) -> ResultadoFPMaxEstrela:
    transacoes = _obter_transacoes(memoria_elite)

    if not transacoes:
        return ResultadoFPMaxEstrela(
            pesos_itens={},
            padroes_maximos=[],
            suporte_minimo=1,
            quantidade_transacoes=0,
        )

    suporte_minimo = _calcular_suporte_minimo(frequencia_minima, len(transacoes))
    padroes_maximos = _minerar_conjuntos_frequentes_maximos(transacoes, suporte_minimo)

    return ResultadoFPMaxEstrela(
        pesos_itens=_calcular_pesos_itens(padroes_maximos),
        padroes_maximos=padroes_maximos,
        suporte_minimo=suporte_minimo,
        quantidade_transacoes=len(transacoes),
    )


def minerar_frequencia(memoria_elite, frequencia_minima=0.3) -> ResultadoFPMaxEstrela:
    """
    Alias mantido para compatibilidade com o restante do projeto.

    A partir daqui, a "mineracao de frequencia" passa a usar FPMax* e retorna
    padroes maximos frequentes, nao apenas contagem de itens individuais.
    """
    return minerar_fpmax_estrela(memoria_elite, frequencia_minima)
