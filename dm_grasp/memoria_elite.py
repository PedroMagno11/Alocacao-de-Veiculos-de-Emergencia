class MemoriaElite:

    def __init__(self, tamanho_maximo):
        self.tamanho_maximo = tamanho_maximo
        self.solucoes = []

    def adicionar(self, solucao, fo):

        self.solucoes.append({
            "fo": fo,
            "solucao": set(solucao)
        })

        self.solucoes.sort(
            key=lambda s: s["fo"],
            reverse=True
        )

        self.solucoes = self.solucoes[
            :self.tamanho_maximo
        ]
    
    def obter_solucoes(self):
        return self.solucoes
    
    def esta_vazia(self):
        return len(self.solucoes) == 0
