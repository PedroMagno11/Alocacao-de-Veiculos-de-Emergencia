from read_instance import ler_instancia, pre_computar_pontos_cobertos


CAMINHO_INSTANCIA = "instancia.csv"

TIPOS_AMBULANCIA = {
    0: {"nome": "B", "raio_cobertura_km": 3.0},
    1: {"nome": "A", "raio_cobertura_km": 5.0},
}


def main():
    dataframe = ler_instancia(CAMINHO_INSTANCIA)

    print("Instancia carregada com sucesso.")
    print(f"Arquivo: {CAMINHO_INSTANCIA}")
    print(f"Quantidade de pontos/regioes: {len(dataframe)}")
    print(f"Colunas: {', '.join(dataframe.columns)}")
    
    
    pontos_cobertos = pre_computar_pontos_cobertos(
        dataframe,
        TIPOS_AMBULANCIA,
    )

    for id_regiao, coberturas_por_tipo in pontos_cobertos.items():
        for tipo, pontos in coberturas_por_tipo.items():
            print(f"P_{id_regiao}^{tipo} = {sorted(pontos)}")

    # return dataframe, pontos_cobertos
    
    
    # 1 - instancia pre-processada e salva em arquivo
    # 2 - validar separadamente construcao e busca local
    # 3 - validação do grasp
    # (obs.:) os códigos dos componentes (construcao, busca local, grasp) podem ser escritos em arquivos separados e importados aqui)

if __name__ == "__main__":
    main()
