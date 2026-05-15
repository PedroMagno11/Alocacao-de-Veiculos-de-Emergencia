import random
import pandas as pd
from pandas import DataFrame
from comum import construir_solucao, busca_local, calcular_score, eh_melhor

def grasp(df: DataFrame, QTD_AMBULANCIAS_A, QTD_AMBULANCIAS_B, ALPHA, TEMPO_COBERTURA, ITERACOES):
    melhor_solucao_global = None
    melhor_avaliacao_global = None

    for iteracao in range(ITERACOES):
        solucao_inicial = construir_solucao(df=df, QTD_AMBULANCIAS_A=QTD_AMBULANCIAS_A, QTD_AMBULANCIAS_B=QTD_AMBULANCIAS_B, ALPHA=ALPHA, TEMPO_COBERTURA=TEMPO_COBERTURA)
        solucao_melhorada = busca_local(df=df, solucao=solucao_inicial, TEMPO_COBERTURA=TEMPO_COBERTURA)

        avaliacao = calcular_score(df=df, solucao=solucao_melhorada, TEMPO_COBERTURA=TEMPO_COBERTURA)

        if eh_melhor(nova_avaliacao=avaliacao, melhor_avaliacao=melhor_avaliacao_global):
            melhor_solucao_global = solucao_melhorada
            melhor_avaliacao_global = avaliacao
        
        print(
            f"Iteração {iteracao + 1} | "
            f"Score: {avaliacao['score_total']:.2f} | "
            f"A: {avaliacao['score_ambulancia_tipo_A']:.2f} | "
            f"B: {avaliacao['score_ambulancia_tipo_B']:.2f} | "
            f"Regiões cobertas: {avaliacao['quant_regioes_cobertas_total']}"
        )

    return melhor_solucao_global, melhor_avaliacao_global

# ----------------------

def imprimir_resultado(melhor_solucao, melhor_avaliacao):
    print("\n==============================")
    print("MELHOR SOLUÇÃO ENCONTRADA")
    print("==============================")

    for local_id, tipo_de_ambulancia in melhor_solucao:
        linha = df.loc[local_id]

        print(
            f"Local {local_id} | "
            f"Ambulância {tipo_de_ambulancia} | "
            f"Complexidade: {linha['complexidade']} | "
            f"Peso: {linha['peso']:.2f} | "
            f"Lat: {linha['latitude']} | "
            f"Lng: {linha['longitude']}"
        )

    print("\n==============================")
    print("AVALIAÇÃO")
    print("==============================")
    print(f"Score total: {melhor_avaliacao['score_total']:.2f}")
    print(f"Score por ambulâncias A: {melhor_avaliacao["score_ambulancia_tipo_A"]:.2f}")
    print(f"Score por ambulâncias B: {melhor_avaliacao["score_ambulancia_tipo_B"]:.2f}")
    print(f"Regiões cobertas no total: {melhor_avaliacao['quant_regioes_cobertas_total']}")


if __name__ == "__main__":
    TEMPO_DE_COBERTURA = 5 # minutos

    QTD_AMBULANCIAS_A = 2
    QTD_AMBULANCIAS_B = 6

    ITERACOES = 5
    ALPHA = 0.3

    random.seed(42)

    df = pd.read_pickle("dados.pkl")
    
    melhor_solucao, melhor_avaliacao = grasp(
        df=df, 
        QTD_AMBULANCIAS_A=QTD_AMBULANCIAS_A, 
        QTD_AMBULANCIAS_B=QTD_AMBULANCIAS_B,
        ALPHA=ALPHA, 
        TEMPO_COBERTURA=TEMPO_DE_COBERTURA,
        ITERACOES=ITERACOES
    )
    imprimir_resultado(melhor_solucao, melhor_avaliacao)