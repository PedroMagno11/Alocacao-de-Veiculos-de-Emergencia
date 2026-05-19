import random
import pandas as pd
from pandas import DataFrame
from multiprocessing import Pool
from collections import defaultdict
from comum import (
    construir_solucao, busca_local, calcular_score,
    eh_melhor, _garantir_cache, _cobertura, _locais_cache
)


# -----------------------------------------------------------------------
# Estrutura de memória
# -----------------------------------------------------------------------

class Memoria:
    """
    Registra, para cada (local_id, tipo_ambulancia), quantas vezes esse
    par apareceu em soluções de alta cobertura, qual a cobertura média
    e qual o score médio nessas soluções.

    Usado na fase guiada para enviesar a construção em direção a locais
    que historicamente maximizam cobertura.
    """

    def __init__(self):
        self.frequencia      = defaultdict(int)    # (local, tipo) → contagem
        self.cobertura_soma  = defaultdict(float)  # (local, tipo) → soma de coberturas
        self.score_soma      = defaultdict(float)  # (local, tipo) → soma de scores
        self.total_solucoes  = 0

    def registrar(self, solucao, avaliacao: dict):
        self.total_solucoes += 1
        cobertura = avaliacao["quant_regioes_cobertas_total"]
        score     = avaliacao["score_total"]

        for local_id, tipo in solucao:
            chave = (local_id, tipo)
            self.frequencia[chave]     += 1
            self.cobertura_soma[chave] += cobertura
            self.score_soma[chave]     += score

    def cobertura_media(self, local_id, tipo) -> float:
        chave = (local_id, tipo)
        freq  = self.frequencia[chave]
        return self.cobertura_soma[chave] / freq if freq > 0 else 0.0

    def score_medio(self, local_id, tipo) -> float:
        chave = (local_id, tipo)
        freq  = self.frequencia[chave]
        return self.score_soma[chave] / freq if freq > 0 else 0.0

    def peso(self, local_id, tipo) -> float:
        """
        Peso usado para enviesar a construção guiada.
        Combina frequência normalizada e cobertura média normalizada.
        Locais nunca vistos recebem peso mínimo (não são excluídos,
        apenas desfavorecidos).
        """
        chave = (local_id, tipo)
        if self.frequencia[chave] == 0:
            return 0.0

        freq_norm     = self.frequencia[chave] / max(self.frequencia.values())
        cob_max       = max(self.cobertura_soma[k] / self.frequencia[k]
                            for k in self.frequencia if self.frequencia[k] > 0)
        cob_norm      = self.cobertura_media(local_id, tipo) / cob_max if cob_max > 0 else 0.0

        # 50% frequência + 50% cobertura média — ajustável via PESO_FREQ
        return 0.5 * freq_norm + 0.5 * cob_norm

    def top_locais(self, tipo: str, n: int = 10) -> list[dict]:
        """Retorna os N locais mais frequentes para um tipo de ambulância."""
        chaves = [k for k in self.frequencia if k[1] == tipo]
        chaves.sort(key=lambda k: self.frequencia[k], reverse=True)

        return [
            {
                "local_id":        k[0],
                "tipo":            k[1],
                "frequencia":      self.frequencia[k],
                "cobertura_media": round(self.cobertura_media(k[0], k[1]), 1),
                "score_medio":     round(self.score_medio(k[0], k[1]), 2),
            }
            for k in chaves[:n]
        ]


# -----------------------------------------------------------------------
# Worker da fase 1 — GRASP normal, devolve solução + avaliação
# -----------------------------------------------------------------------

def _worker_fase1(args: tuple) -> dict:
    df, QTD_A, QTD_B, ALPHA, TEMPO, MAX_ROUNDS, MAX_CAND, seed = args
    random.seed(seed)

    solucao = construir_solucao(df=df, QTD_AMBULANCIAS_A=QTD_A,
                                QTD_AMBULANCIAS_B=QTD_B, ALPHA=ALPHA,
                                TEMPO_COBERTURA=TEMPO)
    solucao = busca_local(df=df, solucao=solucao, TEMPO_COBERTURA=TEMPO,
                          MAX_ROUNDS_SEM_MELHORIA=MAX_ROUNDS,
                          MAX_CANDIDATOS_POR_ROUND=MAX_CAND)
    avaliacao = calcular_score(df=df, solucao=solucao, TEMPO_COBERTURA=TEMPO)

    return {"solucao": solucao, "avaliacao": avaliacao}


# -----------------------------------------------------------------------
# Construção guiada pela memória
# -----------------------------------------------------------------------

def construir_solucao_guiada(
    df: DataFrame,
    QTD_AMBULANCIAS_A: int,
    QTD_AMBULANCIAS_B: int,
    ALPHA: float,
    TEMPO_COBERTURA: int,
    memoria: Memoria,
    FORCA_MEMORIA: float = 0.7,
):
    """
    Variante de construir_solucao que usa a memória para enviesar
    a lista de candidatos restritos.

    FORCA_MEMORIA ∈ [0, 1]:
        0.0 → ignora memória completamente (GRASP puro)
        1.0 → ordena candidatos 100% pelo peso da memória
        0.7 → 70% memória + 30% ganho imediato (padrão)

    O ganho combinado de cada candidato é:
        score_combinado = (1 - FORCA_MEMORIA) * ganho_normalizado
                        + FORCA_MEMORIA       * peso_memoria
    """
    _garantir_cache(df, TEMPO_COBERTURA)

    solucao             = []
    regioes_ja_cobertas = set()

    ambulancias_para_alocar = ["A"] * QTD_AMBULANCIAS_A + ["B"] * QTD_AMBULANCIAS_B
    random.shuffle(ambulancias_para_alocar)

    locais_disponiveis = list(_locais_cache[TEMPO_COBERTURA])

    for tipo_ambulancia in ambulancias_para_alocar:
        col = "demanda_complexa" if tipo_ambulancia == "A" else "demanda_simples"

        # Ganho imediato de cada local
        avaliacoes = []
        for local_id in locais_disponiveis:
            regioes = _cobertura(local_id, TEMPO_COBERTURA)
            novas   = regioes - regioes_ja_cobertas
            ganho   = df.loc[list(novas), col].sum() if novas else 0.0
            avaliacoes.append({
                "local_id":       local_id,
                "tipo_ambulancia": tipo_ambulancia,
                "ganho_imediato": ganho,
                "peso_memoria":   memoria.peso(local_id, tipo_ambulancia),
            })

        # Normaliza ganho imediato para [0, 1]
        max_ganho = max(a["ganho_imediato"] for a in avaliacoes) or 1.0
        for a in avaliacoes:
            a["ganho_norm"] = a["ganho_imediato"] / max_ganho

        # Score combinado
        for a in avaliacoes:
            a["ganho"] = (
                (1 - FORCA_MEMORIA) * a["ganho_norm"]
                + FORCA_MEMORIA     * a["peso_memoria"]
            )

        # Lista de candidatos restritos pelo score combinado
        max_comb  = max(a["ganho"] for a in avaliacoes)
        min_comb  = min(a["ganho"] for a in avaliacoes)
        limite    = max_comb - ALPHA * (max_comb - min_comb)
        restritos = [a for a in avaliacoes if a["ganho"] >= limite]

        escolhido       = random.choice(restritos)
        local_escolhido = escolhido["local_id"]

        solucao.append((local_escolhido, tipo_ambulancia))

        regioes_ja_cobertas.update(_cobertura(local_escolhido, TEMPO_COBERTURA))
        locais_disponiveis.remove(local_escolhido)

    return solucao


# -----------------------------------------------------------------------
# Worker da fase 2 — construção guiada + busca local
# -----------------------------------------------------------------------

def _worker_fase2(args: tuple) -> dict:
    df, QTD_A, QTD_B, ALPHA, TEMPO, MAX_ROUNDS, MAX_CAND, FORCA, memoria, seed = args
    random.seed(seed)

    solucao = construir_solucao_guiada(
        df=df, QTD_AMBULANCIAS_A=QTD_A, QTD_AMBULANCIAS_B=QTD_B,
        ALPHA=ALPHA, TEMPO_COBERTURA=TEMPO,
        memoria=memoria, FORCA_MEMORIA=FORCA,
    )
    solucao = busca_local(df=df, solucao=solucao, TEMPO_COBERTURA=TEMPO,
                          MAX_ROUNDS_SEM_MELHORIA=MAX_ROUNDS,
                          MAX_CANDIDATOS_POR_ROUND=MAX_CAND)
    avaliacao = calcular_score(df=df, solucao=solucao, TEMPO_COBERTURA=TEMPO)

    return {"solucao": solucao, "avaliacao": avaliacao}


# -----------------------------------------------------------------------
# DM-GRASP principal
# -----------------------------------------------------------------------

def dm_grasp(
    df: DataFrame,
    QTD_AMBULANCIAS_A: int,
    QTD_AMBULANCIAS_B: int,
    ALPHA: float,
    TEMPO_COBERTURA: int,
    # Fase 1
    ITERACOES_MEMORIA: int,
    TOP_PORCENTO: float = 0.3,       # fração das melhores soluções por cobertura
    # Fase 2
    ITERACOES_GUIADA: int = 50,
    FORCA_MEMORIA: float = 0.7,
    # Busca local
    MAX_ROUNDS_SEM_MELHORIA: int = 3,
    MAX_CANDIDATOS_POR_ROUND: int = 200,
    # Paralelismo
    NUM_PROCESSOS: int = 4,
    seed_base: int = 42,
) -> tuple[list, dict, Memoria]:

    sep = "─" * 72

    # -------------------------------------------------------------------
    # FASE 1 — construção da memória
    # -------------------------------------------------------------------
    print(sep)
    print(f"  DM-GRASP — FASE 1: MEMÓRIA ({ITERACOES_MEMORIA} iterações)")
    print(f"  Soluções retidas: top {int(TOP_PORCENTO * 100)}% por cobertura")
    print(sep)
    print(f"  {'IT':>4}  {'SCORE TOTAL':>12}  {'TIPO A':>10}  {'TIPO B':>10}  {'REGIÕES':>8}")
    print(sep)

    seeds_f1   = [seed_base + i for i in range(ITERACOES_MEMORIA)]
    solucoes_f1 = []

    with Pool(processes=NUM_PROCESSOS) as pool:
        for inicio in range(0, ITERACOES_MEMORIA, NUM_PROCESSOS):
            lote = seeds_f1[inicio : inicio + NUM_PROCESSOS]
            args = [(df, QTD_AMBULANCIAS_A, QTD_AMBULANCIAS_B, ALPHA,
                     TEMPO_COBERTURA, MAX_ROUNDS_SEM_MELHORIA,
                     MAX_CANDIDATOS_POR_ROUND, s) for s in lote]

            for idx, res in enumerate(pool.map(_worker_fase1, args)):
                it = inicio + idx + 1
                av = res["avaliacao"]
                solucoes_f1.append(res)
                print(
                    f"  {it:>4}  {av['score_total']:>12.2f}  "
                    f"{av['score_ambulancia_tipo_A']:>10.2f}  "
                    f"{av['score_ambulancia_tipo_B']:>10.2f}  "
                    f"{av['quant_regioes_cobertas_total']:>8}"
                )

    # Seleciona o top % por cobertura para alimentar a memória
    solucoes_f1.sort(key=lambda r: r["avaliacao"]["quant_regioes_cobertas_total"], reverse=True)
    n_reter    = max(1, int(len(solucoes_f1) * TOP_PORCENTO))
    top_solucoes = solucoes_f1[:n_reter]

    memoria = Memoria()
    for r in top_solucoes:
        memoria.registrar(r["solucao"], r["avaliacao"])

    # Imprime top 10 por tipo
    print(sep)
    print("  PADRÕES APRENDIDOS — Top 10 locais por tipo")
    print(sep)

    for tipo in ["A", "B"]:
        print(f"\n  Ambulância tipo {tipo}:")
        print(f"  {'LOCAL':>8}  {'FREQ':>6}  {'COB. MÉDIA':>12}  {'SCORE MÉDIO':>13}")
        for entry in memoria.top_locais(tipo=tipo, n=10):
            print(
                f"  {entry['local_id']:>8}  {entry['frequencia']:>6}  "
                f"{entry['cobertura_media']:>12.1f}  {entry['score_medio']:>13.2f}"
            )

    # -------------------------------------------------------------------
    # FASE 2 — construção guiada pela memória
    # -------------------------------------------------------------------
    print(f"\n{sep}")
    print(f"  DM-GRASP — FASE 2: GUIADA ({ITERACOES_GUIADA} iterações | força memória={FORCA_MEMORIA})")
    print(sep)
    print(f"  {'IT':>4}  {'SCORE TOTAL':>12}  {'TIPO A':>10}  {'TIPO B':>10}  {'REGIÕES':>8}")
    print(sep)

    melhor_solucao_global   = None
    melhor_avaliacao_global = None

    seeds_f2 = [seed_base + ITERACOES_MEMORIA + i for i in range(ITERACOES_GUIADA)]

    with Pool(processes=NUM_PROCESSOS) as pool:
        for inicio in range(0, ITERACOES_GUIADA, NUM_PROCESSOS):
            lote = seeds_f2[inicio : inicio + NUM_PROCESSOS]
            args = [(df, QTD_AMBULANCIAS_A, QTD_AMBULANCIAS_B, ALPHA,
                     TEMPO_COBERTURA, MAX_ROUNDS_SEM_MELHORIA,
                     MAX_CANDIDATOS_POR_ROUND, FORCA_MEMORIA, memoria, s)
                    for s in lote]

            for idx, res in enumerate(pool.map(_worker_fase2, args)):
                it = inicio + idx + 1
                av = res["avaliacao"]
                so = res["solucao"]

                nova_melhor = eh_melhor(av, melhor_avaliacao_global)
                if nova_melhor:
                    melhor_solucao_global   = so
                    melhor_avaliacao_global = av

                flag = "  ◄ MELHOR" if nova_melhor else ""
                print(
                    f"  {it:>4}  {av['score_total']:>12.2f}  "
                    f"{av['score_ambulancia_tipo_A']:>10.2f}  "
                    f"{av['score_ambulancia_tipo_B']:>10.2f}  "
                    f"{av['quant_regioes_cobertas_total']:>8}{flag}"
                )

    print(sep)
    return melhor_solucao_global, melhor_avaliacao_global, memoria


# -----------------------------------------------------------------------
# Impressão e salvamento
# -----------------------------------------------------------------------

def imprimir_resultado(df: DataFrame, melhor_solucao, melhor_avaliacao):
    sep = "─" * 72
    print(f"\n{sep}")
    print("  MELHOR SOLUÇÃO ENCONTRADA")
    print(sep)
    for local_id, tipo in melhor_solucao:
        linha = df.loc[local_id]
        print(
            f"  Local {local_id:<6} | Ambulância {tipo} | "
            f"Complexidade: {linha['complexidade']:<8} | "
            f"Peso: {linha['peso']:>6.2f} | "
            f"Lat: {linha['latitude']} | Lng: {linha['longitude']}"
        )
    print(sep)
    print("  AVALIAÇÃO FINAL")
    print(sep)
    print(f"  Score total:          {melhor_avaliacao['score_total']:.2f}")
    print(f"  Score ambulâncias A:  {melhor_avaliacao['score_ambulancia_tipo_A']:.2f}")
    print(f"  Score ambulâncias B:  {melhor_avaliacao['score_ambulancia_tipo_B']:.2f}")
    print(f"  Regiões cobertas:     {melhor_avaliacao['quant_regioes_cobertas_total']}")
    print(sep)


def salvar_resultado(df: DataFrame, melhor_solucao, melhor_avaliacao,
                     memoria: Memoria, caminho: str = "dm_resultado"):
    import csv
    from datetime import datetime

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Alocações da melhor solução
    with open(f"{caminho}_alocacoes_{ts}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["local_id", "tipo_ambulancia", "complexidade", "peso", "latitude", "longitude"])
        for local_id, tipo in melhor_solucao:
            l = df.loc[local_id]
            w.writerow([local_id, tipo, l["complexidade"], l["peso"], l["latitude"], l["longitude"]])

    # Avaliação final
    with open(f"{caminho}_avaliacao_{ts}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metrica", "valor"])
        w.writerow(["score_total",                  melhor_avaliacao["score_total"]])
        w.writerow(["score_ambulancias_A",          melhor_avaliacao["score_ambulancia_tipo_A"]])
        w.writerow(["score_ambulancias_B",          melhor_avaliacao["score_ambulancia_tipo_B"]])
        w.writerow(["quant_regioes_cobertas_total", melhor_avaliacao["quant_regioes_cobertas_total"]])

    # Padrões da memória — top por tipo
    with open(f"{caminho}_padroes_{ts}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tipo", "local_id", "frequencia", "cobertura_media", "score_medio"])
        for tipo in ["A", "B"]:
            for entry in memoria.top_locais(tipo=tipo, n=10):
                w.writerow([
                    entry["tipo"], entry["local_id"], entry["frequencia"],
                    entry["cobertura_media"], entry["score_medio"],
                ])

    print(f"  Arquivos salvos com prefixo: {caminho}_*_{ts}.csv")


# -----------------------------------------------------------------------

if __name__ == "__main__":
    TEMPO_DE_COBERTURA       = 5
    QTD_AMBULANCIAS_A        = 1
    QTD_AMBULANCIAS_B        = 4
    ALPHA                    = 0.3
    NUM_PROCESSOS            = 4

    # Fase 1
    ITERACOES_MEMORIA        = 60    # iterações para aprender padrões
    TOP_PORCENTO             = 0.3   # top 30% por cobertura entram na memória

    # Fase 2
    ITERACOES_GUIADA         = 40    # iterações guiadas pela memória
    FORCA_MEMORIA            = 0.7   # 0=GRASP puro, 1=100% guiado pela memória

    # Busca local
    MAX_ROUNDS_SEM_MELHORIA  = 3
    MAX_CANDIDATOS_POR_ROUND = 200

    df = pd.read_pickle("dados.pkl")

    melhor_solucao, melhor_avaliacao, memoria = dm_grasp(
        df=df,
        QTD_AMBULANCIAS_A=QTD_AMBULANCIAS_A,
        QTD_AMBULANCIAS_B=QTD_AMBULANCIAS_B,
        ALPHA=ALPHA,
        TEMPO_COBERTURA=TEMPO_DE_COBERTURA,
        ITERACOES_MEMORIA=ITERACOES_MEMORIA,
        TOP_PORCENTO=TOP_PORCENTO,
        ITERACOES_GUIADA=ITERACOES_GUIADA,
        FORCA_MEMORIA=FORCA_MEMORIA,
        MAX_ROUNDS_SEM_MELHORIA=MAX_ROUNDS_SEM_MELHORIA,
        MAX_CANDIDATOS_POR_ROUND=MAX_CANDIDATOS_POR_ROUND,
        NUM_PROCESSOS=NUM_PROCESSOS,
        seed_base=42,
    )

    imprimir_resultado(df=df, melhor_solucao=melhor_solucao, melhor_avaliacao=melhor_avaliacao)
    salvar_resultado(df=df, melhor_solucao=melhor_solucao, melhor_avaliacao=melhor_avaliacao,
                     memoria=memoria, caminho="dm_resultado")
