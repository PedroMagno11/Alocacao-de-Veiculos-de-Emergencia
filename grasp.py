import random
import pandas as pd
from pandas import DataFrame
from multiprocessing import Pool
from comum import construir_solucao, busca_local, calcular_score, eh_melhor


# -----------------------------------------------------------------------
# Função executada por cada processo worker
# -----------------------------------------------------------------------

def _executar_iteracao(args: tuple) -> dict:
    df, QTD_AMBULANCIAS_A, QTD_AMBULANCIAS_B, ALPHA, TEMPO_COBERTURA, MAX_ROUNDS_SEM_MELHORIA, MAX_CANDIDATOS_POR_ROUND, seed = args

    random.seed(seed)

    solucao_inicial = construir_solucao(
        df=df,
        QTD_AMBULANCIAS_A=QTD_AMBULANCIAS_A,
        QTD_AMBULANCIAS_B=QTD_AMBULANCIAS_B,
        ALPHA=ALPHA,
        TEMPO_COBERTURA=TEMPO_COBERTURA
    )

    solucao_melhorada = busca_local(
        df=df,
        solucao=solucao_inicial,
        TEMPO_COBERTURA=TEMPO_COBERTURA,
        MAX_ROUNDS_SEM_MELHORIA=MAX_ROUNDS_SEM_MELHORIA,
        MAX_CANDIDATOS_POR_ROUND=MAX_CANDIDATOS_POR_ROUND,
    )

    avaliacao = calcular_score(
        df=df,
        solucao=solucao_melhorada,
        TEMPO_COBERTURA=TEMPO_COBERTURA
    )

    return {
        "solucao":   solucao_melhorada,
        "avaliacao": avaliacao,
    }


# -----------------------------------------------------------------------
# GRASP paralelizado
# -----------------------------------------------------------------------

def grasp(
    df: DataFrame,
    QTD_AMBULANCIAS_A: int,
    QTD_AMBULANCIAS_B: int,
    ALPHA: float,
    TEMPO_COBERTURA: int,
    ITERACOES: int,
    MAX_ROUNDS_SEM_MELHORIA: int = None,
    MAX_CANDIDATOS_POR_ROUND: int = None,
    NUM_PROCESSOS: int = 4,
    seed_base: int = 42,
):
    melhor_solucao_global   = None
    melhor_avaliacao_global = None

    seeds = [seed_base + i for i in range(ITERACOES)]

    sep = "─" * 72
    limite_label = str(MAX_ROUNDS_SEM_MELHORIA) if MAX_ROUNDS_SEM_MELHORIA is not None else "∞ (convergência total)"
    cand_label   = str(MAX_CANDIDATOS_POR_ROUND) if MAX_CANDIDATOS_POR_ROUND is not None else "todos"

    print(sep)
    print(f"  GRASP | {ITERACOES} iterações | {NUM_PROCESSOS} processos | α={ALPHA} | cobertura={TEMPO_COBERTURA} min")
    print(f"  Busca local: para após {limite_label} rounds sem melhoria | candidatos/round: {cand_label}")
    print(sep)
    print(f"  {'IT':>4}  {'SCORE TOTAL':>12}  {'TIPO A':>10}  {'TIPO B':>10}  {'REGIÕES':>8}")
    print(sep)

    iteracao_atual = 0

    with Pool(processes=NUM_PROCESSOS) as pool:
        for inicio in range(0, ITERACOES, NUM_PROCESSOS):
            lote_seeds = seeds[inicio : inicio + NUM_PROCESSOS]

            argumentos = [
                (df, QTD_AMBULANCIAS_A, QTD_AMBULANCIAS_B, ALPHA, TEMPO_COBERTURA, MAX_ROUNDS_SEM_MELHORIA, MAX_CANDIDATOS_POR_ROUND, s)
                for s in lote_seeds
            ]

            resultados = pool.map(_executar_iteracao, argumentos)

            for resultado in resultados:
                iteracao_atual += 1
                avaliacao = resultado["avaliacao"]
                solucao   = resultado["solucao"]

                nova_melhor = eh_melhor(
                    nova_avaliacao=avaliacao,
                    melhor_avaliacao=melhor_avaliacao_global
                )

                if nova_melhor:
                    melhor_solucao_global   = solucao
                    melhor_avaliacao_global = avaliacao

                flag = "  ◄ MELHOR" if nova_melhor else ""

                print(
                    f"  {iteracao_atual:>4}  "
                    f"{avaliacao['score_total']:>12.2f}  "
                    f"{avaliacao['score_ambulancia_tipo_A']:>10.2f}  "
                    f"{avaliacao['score_ambulancia_tipo_B']:>10.2f}  "
                    f"{avaliacao['quant_regioes_cobertas_total']:>8}"
                    f"{flag}"
                )

    print(sep)

    return melhor_solucao_global, melhor_avaliacao_global


# -----------------------------------------------------------------------
# Impressão do resultado final
# -----------------------------------------------------------------------

def salvar_resultado(df: DataFrame, melhor_solucao, melhor_avaliacao, caminho: str = "resultado"):
    """
    Salva a melhor solução em dois CSVs:
      - {caminho}_alocacoes.csv  → uma linha por ambulância alocada
      - {caminho}_avaliacao.csv  → scores e regiões cobertas
    """
    import csv
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    arq_alocacoes = f"{caminho}_alocacoes_{timestamp}.csv"
    arq_avaliacao = f"{caminho}_avaliacao_{timestamp}.csv"

    with open(arq_alocacoes, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["local_id", "tipo_ambulancia", "complexidade", "peso", "latitude", "longitude"])
        for local_id, tipo_de_ambulancia in melhor_solucao:
            linha = df.loc[local_id]
            writer.writerow([
                local_id,
                tipo_de_ambulancia,
                linha["complexidade"],
                linha["peso"],
                linha["latitude"],
                linha["longitude"],
            ])

    with open(arq_avaliacao, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metrica", "valor"])
        writer.writerow(["score_total",                  melhor_avaliacao["score_total"]])
        writer.writerow(["score_ambulancias_A",          melhor_avaliacao["score_ambulancia_tipo_A"]])
        writer.writerow(["score_ambulancias_B",          melhor_avaliacao["score_ambulancia_tipo_B"]])
        writer.writerow(["quant_regioes_cobertas_total", melhor_avaliacao["quant_regioes_cobertas_total"]])

    print(f"  Alocações salvas em: {arq_alocacoes}")
    print(f"  Avaliação salva em:  {arq_avaliacao}")


def imprimir_resultado(df: DataFrame, melhor_solucao, melhor_avaliacao):
    sep = "─" * 72

    print("\n" + sep)
    print("  MELHOR SOLUÇÃO ENCONTRADA")
    print(sep)

    for local_id, tipo_de_ambulancia in melhor_solucao:
        linha = df.loc[local_id]
        print(
            f"  Local {local_id:<6} | "
            f"Ambulância {tipo_de_ambulancia} | "
            f"Complexidade: {linha['complexidade']:<8} | "
            f"Peso: {linha['peso']:>6.2f} | "
            f"Lat: {linha['latitude']} | "
            f"Lng: {linha['longitude']}"
        )

    print(sep)
    print("  AVALIAÇÃO FINAL")
    print(sep)
    print(f"  Score total:          {melhor_avaliacao['score_total']:.2f}")
    print(f"  Score ambulâncias A:  {melhor_avaliacao['score_ambulancia_tipo_A']:.2f}")
    print(f"  Score ambulâncias B:  {melhor_avaliacao['score_ambulancia_tipo_B']:.2f}")
    print(f"  Regiões cobertas:     {melhor_avaliacao['quant_regioes_cobertas_total']}")
    print(sep)


# -----------------------------------------------------------------------

if __name__ == "__main__":
    TEMPO_DE_COBERTURA       = 5
    QTD_AMBULANCIAS_A        = 1
    QTD_AMBULANCIAS_B        = 4
    ITERACOES                = 100
    ALPHA                    = 0.3
    NUM_PROCESSOS            = 4
    MAX_ROUNDS_SEM_MELHORIA  = 3    # None = convergência total
    MAX_CANDIDATOS_POR_ROUND = 200  # None = avalia todos (~3011); 100-300 é um bom intervalo

    df = pd.read_pickle("dados.pkl")

    melhor_solucao, melhor_avaliacao = grasp(
        df=df,
        QTD_AMBULANCIAS_A=QTD_AMBULANCIAS_A,
        QTD_AMBULANCIAS_B=QTD_AMBULANCIAS_B,
        ALPHA=ALPHA,
        TEMPO_COBERTURA=TEMPO_DE_COBERTURA,
        ITERACOES=ITERACOES,
        MAX_ROUNDS_SEM_MELHORIA=MAX_ROUNDS_SEM_MELHORIA,
        MAX_CANDIDATOS_POR_ROUND=MAX_CANDIDATOS_POR_ROUND,
        NUM_PROCESSOS=NUM_PROCESSOS,
        seed_base=42,
    )

    imprimir_resultado(df=df, melhor_solucao=melhor_solucao, melhor_avaliacao=melhor_avaliacao)
    salvar_resultado(df=df, melhor_solucao=melhor_solucao, melhor_avaliacao=melhor_avaliacao, caminho="resultado")
