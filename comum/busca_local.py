"""
Busca local e funcoes de avaliacao de solucoes para o PMCS-FA usando bitsets.

Esta versao respeita a formulacao do PDF:

    max sum |P_i^t| x_i^t
        - sum |P_i^t inter P_j^t'| y_ij^tt', com j != i

A otimizacao principal e representar cada conjunto P_i^t como uma mascara
binaria inteira. Assim:

    |P_i^t|              -> mascara.bit_count()
    |P_i^t inter P_j^t'| -> (mascara_i & mascara_j).bit_count()
    uniao fisica         -> mascara_a | mascara_b | ...

Isso evita intersecoes de set() durante a busca local. Tambem usamos
first improvement: ao encontrar a primeira melhora, o movimento e aceito e a
exploracao reinicia da primeira vizinhanca.
"""

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

Alocacao = Tuple[int, str]
PontosCobertos = Dict[int, Dict[str, FrozenSet[int]]]


@dataclass
class PreCalculoBitset:
    """
    Estrutura em memoria para consulta rapida de cobertura e intersecoes.

    mascara[(i, t)] = bitset dos pontos cobertos por colocar uma ambulancia
    do tipo t na regiao i.

    area[(i, t)] = quantidade de bits ligados em mascara[(i, t)].

    cache_intersecao guarda cardinalidades ja consultadas durante a busca.
    Isso evita montar uma matriz |R|^2 |T|^2 gigantesca quando a busca local
    usa apenas uma parte dos pares.
    """

    mascara: Dict[Alocacao, int]
    area: Dict[Alocacao, int]
    ponto_para_indice: Dict[int, int]
    indice_para_ponto: Dict[int, int]
    cache_intersecao: Dict[Tuple[Alocacao, Alocacao], int] = field(default_factory=dict)

    def obter_area(self, alocacao: Alocacao) -> int:
        return self.area.get(alocacao, 0)

    def obter_mascara(self, alocacao: Alocacao) -> int:
        return self.mascara.get(alocacao, 0)

    def obter_intersecao(self, primeira: Alocacao, segunda: Alocacao) -> int:
        if primeira == segunda or primeira[0] == segunda[0]:
            # A restricao do problema impede duas ambulancias na mesma regiao,
            # e a formulacao considera j != i na penalizacao.
            return 0

        chave = (primeira, segunda)
        valor = self.cache_intersecao.get(chave)
        if valor is not None:
            return valor

        valor = (self.obter_mascara(primeira) & self.obter_mascara(segunda)).bit_count()
        self.cache_intersecao[chave] = valor
        return valor

    def mascara_da_solucao(self, solucao: Iterable[Alocacao]) -> int:
        mascara_total = 0
        for alocacao in solucao:
            mascara_total |= self.obter_mascara(alocacao)
        return mascara_total

    def pontos_da_mascara(self, mascara: int) -> Set[int]:
        pontos: Set[int] = set()
        indice = 0
        while mascara:
            if mascara & 1:
                pontos.add(self.indice_para_ponto[indice])
            mascara >>= 1
            indice += 1
        return pontos


def _criar_mapeamento_pontos(pontos_cobertos: PontosCobertos) -> Tuple[Dict[int, int], Dict[int, int]]:
    todos_os_pontos: Set[int] = set()
    for coberturas_por_tipo in pontos_cobertos.values():
        for pontos in coberturas_por_tipo.values():
            todos_os_pontos.update(int(ponto) for ponto in pontos)

    pontos_ordenados = sorted(todos_os_pontos)
    ponto_para_indice = {ponto: indice for indice, ponto in enumerate(pontos_ordenados)}
    indice_para_ponto = {indice: ponto for ponto, indice in ponto_para_indice.items()}
    return ponto_para_indice, indice_para_ponto


def _converter_pontos_para_mascara(pontos: Iterable[int], ponto_para_indice: Dict[int, int]) -> int:
    mascara = 0
    for ponto in pontos:
        mascara |= 1 << ponto_para_indice[int(ponto)]
    return mascara


def pre_computar_bitsets(
    pontos_cobertos: PontosCobertos,
    regioes: Iterable[int],
    tipos_ambulancia: Iterable[str],
    pre_computar_todas_intersecoes: bool = False,
) -> PreCalculoBitset:
    """
    Pre-computa as coberturas como bitsets em memoria.

    Por padrao, as intersecoes sao calculadas sob demanda e guardadas em cache,
    que costuma ser melhor para memoria. Se quiser forcar a matriz completa de
    intersecoes, passe pre_computar_todas_intersecoes=True.
    """
    ponto_para_indice, indice_para_ponto = _criar_mapeamento_pontos(pontos_cobertos)

    candidatos: List[Alocacao] = [
        (int(id_regiao), tipo)
        for id_regiao in regioes
        for tipo in tipos_ambulancia
    ]

    mascara: Dict[Alocacao, int] = {}
    area: Dict[Alocacao, int] = {}

    for alocacao in candidatos:
        id_regiao, tipo = alocacao
        mascara_alocacao = _converter_pontos_para_mascara(
            pontos_cobertos[id_regiao][tipo],
            ponto_para_indice,
        )
        mascara[alocacao] = mascara_alocacao
        area[alocacao] = mascara_alocacao.bit_count()

    pre_calculo = PreCalculoBitset(
        mascara=mascara,
        area=area,
        ponto_para_indice=ponto_para_indice,
        indice_para_ponto=indice_para_ponto,
    )

    if pre_computar_todas_intersecoes:
        for primeira in candidatos:
            for segunda in candidatos:
                if primeira == segunda or primeira[0] == segunda[0]:
                    continue
                pre_calculo.cache_intersecao[(primeira, segunda)] = (
                    mascara[primeira] & mascara[segunda]
                ).bit_count()

    return pre_calculo


def obter_candidatos_vizinhanca_completa(
    regioes: Iterable[int],
    tipos_ambulancia: Iterable[str],
) -> List[Alocacao]:
    """
    Retorna todos os candidatos (i, t) da vizinhanca, sem amostragem, sem corte
    e sem ordenacao por heuristica.

    A ordem e deterministica: percorre os tipos na ordem recebida e, para cada
    tipo, percorre todas as regioes na ordem recebida. Isso garante que a busca
    local examine a vizinhanca completa ate encontrar a primeira melhora.
    """
    return [(int(id_regiao), tipo) for tipo in tipos_ambulancia for id_regiao in regioes]


# Alias legado. Mantido para compatibilidade, mas agora NAO ordena por area.
def obter_candidatos_ordenados_por_area(
    regioes: Iterable[int],
    tipos_ambulancia: Iterable[str],
    pre_calculo: PreCalculoBitset,
) -> List[Alocacao]:
    return obter_candidatos_vizinhanca_completa(regioes, tipos_ambulancia)


# Alias mantido para compatibilidade com a versao anterior.
def pre_computar_areas_e_intersecoes(
    pontos_cobertos: PontosCobertos,
    regioes: Iterable[int],
    tipos_ambulancia: Iterable[str],
) -> PreCalculoBitset:
    return pre_computar_bitsets(
        pontos_cobertos=pontos_cobertos,
        regioes=regioes,
        tipos_ambulancia=tipos_ambulancia,
        pre_computar_todas_intersecoes=False,
    )


def obter_pontos_cobertos_pela_solucao(
    solucao: Iterable[Alocacao],
    pontos_cobertos: Optional[PontosCobertos] = None,
    pre_calculo: Optional[PreCalculoBitset] = None,
) -> Set[int]:
    """
    Retorna a uniao fisica dos pontos cobertos por pelo menos uma alocacao.
    Preferencialmente use pre_calculo, pois a uniao por bitset e mais rapida.
    """
    if pre_calculo is not None:
        return pre_calculo.pontos_da_mascara(pre_calculo.mascara_da_solucao(solucao))

    if pontos_cobertos is None:
        raise ValueError("Informe pontos_cobertos ou pre_calculo.")

    cobertura_total: Set[int] = set()
    for id_regiao, tipo in solucao:
        cobertura_total |= set(pontos_cobertos[id_regiao][tipo])
    return cobertura_total


def contar_ambulancias_por_tipo(solucao: Iterable[Alocacao]) -> Dict[str, int]:
    contagem: Dict[str, int] = {}
    for _, tipo in solucao:
        contagem[tipo] = contagem.get(tipo, 0) + 1
    return contagem


def calcular_funcao_objetivo(
    solucao: Iterable[Alocacao],
    pontos_cobertos: Optional[PontosCobertos] = None,
    pre_calculo: Optional[PreCalculoBitset] = None,
) -> float:
    """Calcula a FO do PMCS-FA conforme cobertura individual - intersecoes."""
    solucao_lista = list(solucao)
    if not solucao_lista:
        return 0.0

    if pre_calculo is None:
        if pontos_cobertos is None:
            raise ValueError("Informe pontos_cobertos ou pre_calculo.")
        regioes = sorted({id_regiao for id_regiao, _ in solucao_lista})
        tipos = sorted({tipo for _, tipo in solucao_lista})
        pre_calculo = pre_computar_bitsets(pontos_cobertos, regioes, tipos)

    cobertura_individual = sum(pre_calculo.obter_area(alocacao) for alocacao in solucao_lista)

    penalizacao_sobreposicao = 0
    for primeira in solucao_lista:
        for segunda in solucao_lista:
            if primeira == segunda:
                continue
            penalizacao_sobreposicao += pre_calculo.obter_intersecao(primeira, segunda)

    return float(cobertura_individual - penalizacao_sobreposicao)


def _delta_insercao(
    candidato: Alocacao,
    solucao_atual: Set[Alocacao],
    pre_calculo: PreCalculoBitset,
) -> int:
    penalizacao_nova = 0
    for alocacao in solucao_atual:
        penalizacao_nova += pre_calculo.obter_intersecao(candidato, alocacao)
        penalizacao_nova += pre_calculo.obter_intersecao(alocacao, candidato)
    return pre_calculo.obter_area(candidato) - penalizacao_nova


def _delta_remocao(
    alocacao_removida: Alocacao,
    solucao_atual: Set[Alocacao],
    pre_calculo: PreCalculoBitset,
) -> int:
    penalizacao_removida = 0
    for alocacao in solucao_atual:
        if alocacao == alocacao_removida:
            continue
        penalizacao_removida += pre_calculo.obter_intersecao(alocacao_removida, alocacao)
        penalizacao_removida += pre_calculo.obter_intersecao(alocacao, alocacao_removida)
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
    candidatos_vizinhanca: Sequence[Alocacao],
    quantidade_maxima_por_tipo: Dict[str, int],
    pre_calculo: PreCalculoBitset,
) -> Optional[Tuple[Set[Alocacao], int]]:
    """
    Percorre TODA a vizinhanca de troca em ordem deterministica e aplica
    first improvement.

    Isto significa: testa todas as remocoes possiveis e, para cada uma, todos
    os candidatos de insercao viaveis. Ao encontrar o primeiro delta positivo,
    aceita o movimento imediatamente e devolve a nova solucao. A busca principal
    entao reinicia a exploracao desde a primeira vizinhanca.
    """
    for alocacao_removida in list(solucao_atual):
        solucao_sem = set(solucao_atual)
        solucao_sem.remove(alocacao_removida)
        regioes_ocupadas = {id_regiao for id_regiao, _ in solucao_sem}
        contagem_tipo = contar_ambulancias_por_tipo(solucao_sem)
        delta_remocao = _delta_remocao(alocacao_removida, solucao_atual, pre_calculo)

        for candidato in candidatos_vizinhanca:
            if candidato == alocacao_removida:
                continue
            if not _eh_viavel_para_insercao(
                candidato,
                regioes_ocupadas,
                contagem_tipo,
                quantidade_maxima_por_tipo,
            ):
                continue

            # Poda simples: se nem a area bruta do candidato compensa a remocao,
            # o delta real, que ainda desconta sobreposicoes, nao sera positivo.
            if delta_remocao + pre_calculo.obter_area(candidato) <= 0:
                continue

            delta_total = delta_remocao + _delta_insercao(candidato, solucao_sem, pre_calculo)
            if delta_total > 0:
                nova_solucao = set(solucao_sem)
                nova_solucao.add(candidato)
                return nova_solucao, delta_total
    return None

def _tentar_primeira_melhora_insercao(
    solucao_atual: Set[Alocacao],
    candidatos_vizinhanca: Sequence[Alocacao],
    quantidade_maxima_por_tipo: Dict[str, int],
    pre_calculo: PreCalculoBitset,
) -> Optional[Tuple[Set[Alocacao], int]]:
    """Percorre TODA a vizinhanca de insercao e aceita a primeira melhora."""
    regioes_ocupadas = {id_regiao for id_regiao, _ in solucao_atual}
    contagem_tipo = contar_ambulancias_por_tipo(solucao_atual)

    for candidato in candidatos_vizinhanca:
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
    pre_calculo: PreCalculoBitset,
) -> Optional[Tuple[Set[Alocacao], int]]:
    """Percorre TODA a vizinhanca de remocao e aceita a primeira melhora."""
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
    pre_calculo: Optional[PreCalculoBitset] = None,
):
    """
    Busca local com first improvement e avaliacao incremental por bitset.

    A vizinhanca NAO e reduzida nem amostrada. A cada iteracao, o algoritmo
    percorre a vizinhanca completa na ordem definida e aceita a primeira melhora
    encontrada. Depois de aceitar uma melhora, reinicia a busca pela primeira
    vizinhanca.

    Vizinhancas, nesta ordem:
      1. troca;
      2. insercao;
      3. remocao.
    """
    if pre_calculo is None:
        pre_calculo = pre_computar_bitsets(
            pontos_cobertos,
            regioes,
            tipos_ambulancia,
        )

    melhor_solucao: Set[Alocacao] = set(solucao_inicial)
    melhor_fo = calcular_funcao_objetivo(melhor_solucao, pre_calculo=pre_calculo)

    candidatos_vizinhanca = obter_candidatos_vizinhanca_completa(
        regioes,
        tipos_ambulancia,
    )

    iteracao = 0
    limite_iteracoes = float("inf") if max_iteracoes_sem_melhora is None else max_iteracoes_sem_melhora

    while iteracao < limite_iteracoes:
        movimento = (
            _tentar_primeira_melhora_troca(
                melhor_solucao,
                candidatos_vizinhanca,
                quantidade_maxima_por_tipo,
                pre_calculo,
            )
            or _tentar_primeira_melhora_insercao(
                melhor_solucao,
                candidatos_vizinhanca,
                quantidade_maxima_por_tipo,
                pre_calculo,
            )
            or _tentar_primeira_melhora_remocao(melhor_solucao, pre_calculo)
        )

        # Em first improvement, se nenhuma vizinhanca encontrou melhora,
        # chegamos a um otimo local para essas vizinhancas. Deve parar.
        # A versao anterior incrementava iteracoes_sem_melhora e podia ficar
        # em loop infinito quando max_iteracoes_sem_melhora era None.
        if movimento is None:
            break

        melhor_solucao, delta = movimento
        melhor_fo += delta
        iteracao += 1

    return melhor_solucao, float(melhor_fo)
