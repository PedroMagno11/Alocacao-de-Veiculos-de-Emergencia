import tkinter as tk
from collections import deque

TAMANHO_CELULA = 30
LINHAS = 30
COLUNAS = 30

COR_LIVRE = "white"
COR_OBSTACULO = "black"
COR_INICIO = "green"
COR_FIM = "red"
COR_ROTA = "yellow"
COR_GRID = "gray"


def bfs(mapa, inicio, fim):
    if inicio is None or fim is None:
        return []

    linhas = len(mapa)
    colunas = len(mapa[0])

    if mapa[inicio[0]][inicio[1]] == 1 or mapa[fim[0]][fim[1]] == 1:
        return []

    fila = deque([inicio])
    pais = {inicio: None}
    movimentos = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    while fila:
        atual = fila.popleft()

        if atual == fim:
            break

        for dl, dc in movimentos:
            vizinho = (atual[0] + dl, atual[1] + dc)

            if 0 <= vizinho[0] < linhas and 0 <= vizinho[1] < colunas:
                if mapa[vizinho[0]][vizinho[1]] == 0 and vizinho not in pais:
                    fila.append(vizinho)
                    pais[vizinho] = atual

    if fim not in pais:
        return []

    caminho = []
    passo = fim
    while passo is not None:
        caminho.append(passo)
        passo = pais[passo]

    caminho.reverse()
    return caminho


class EditorMapa:
    def __init__(self, root):
        self.root = root
        self.root.title("Editor de mapa com rota em tempo real")

        self.mapa = [[0 for _ in range(COLUNAS)] for _ in range(LINHAS)]
        self.inicio = None
        self.fim = None
        self.rota = []

        self.modo = "obstaculo"
        self.botao_mouse = None

        self.label_info = tk.Label(
            root,
            text="Modo: obstaculo | Teclas: i=início, f=fim, c=limpar, esc=normal",
            anchor="w"
        )
        self.label_info.pack(fill="x")

        self.canvas = tk.Canvas(
            root,
            width=COLUNAS * TAMANHO_CELULA,
            height=LINHAS * TAMANHO_CELULA,
            bg="white"
        )
        self.canvas.pack()

        self.canvas.bind("<Button-1>", self.on_button_1)
        self.canvas.bind("<B1-Motion>", self.on_drag_1)

        self.canvas.bind("<Button-3>", self.on_button_3)
        self.canvas.bind("<B3-Motion>", self.on_drag_3)

        self.root.bind("<Key>", self.on_key_press)

        self.desenhar()

    def limpar(self):
        self.mapa = [[0 for _ in range(COLUNAS)] for _ in range(LINHAS)]
        self.inicio = None
        self.fim = None
        self.rota = []
        self.atualizar_status()
        self.desenhar()

    def atualizar_status(self):
        self.label_info.config(
            text=f"Modo: {self.modo} | Início: {self.inicio} | Fim: {self.fim} | Passos: {max(len(self.rota)-1, 0)} | Teclas: i, f, c, esc"
        )

    def pos_mouse_para_celula(self, event):
        coluna = event.x // TAMANHO_CELULA
        linha = event.y // TAMANHO_CELULA

        if 0 <= linha < LINHAS and 0 <= coluna < COLUNAS:
            return linha, coluna
        return None

    def recalcular_rota(self):
        self.rota = bfs(self.mapa, self.inicio, self.fim)

    def aplicar_acao(self, linha, coluna, botao):
        if self.modo == "inicio":
            if self.fim == (linha, coluna):
                self.fim = None
            self.mapa[linha][coluna] = 0
            self.inicio = (linha, coluna)
            self.modo = "obstaculo"
            self.recalcular_rota()
            self.atualizar_status()
            self.desenhar()
            return

        if self.modo == "fim":
            if self.inicio == (linha, coluna):
                self.inicio = None
            self.mapa[linha][coluna] = 0
            self.fim = (linha, coluna)
            self.modo = "obstaculo"
            self.recalcular_rota()
            self.atualizar_status()
            self.desenhar()
            return

        if botao == 1:
            if self.inicio != (linha, coluna) and self.fim != (linha, coluna):
                self.mapa[linha][coluna] = 1
        elif botao == 3:
            self.mapa[linha][coluna] = 0

        self.recalcular_rota()
        self.atualizar_status()
        self.desenhar()

    def on_button_1(self, event):
        pos = self.pos_mouse_para_celula(event)
        if pos:
            self.aplicar_acao(pos[0], pos[1], 1)

    def on_drag_1(self, event):
        pos = self.pos_mouse_para_celula(event)
        if pos:
            self.aplicar_acao(pos[0], pos[1], 1)

    def on_button_3(self, event):
        pos = self.pos_mouse_para_celula(event)
        if pos:
            self.aplicar_acao(pos[0], pos[1], 3)

    def on_drag_3(self, event):
        pos = self.pos_mouse_para_celula(event)
        if pos:
            self.aplicar_acao(pos[0], pos[1], 3)

    def on_key_press(self, event):
        tecla = event.keysym.lower()

        if tecla == "i":
            self.modo = "inicio"
        elif tecla == "f":
            self.modo = "fim"
        elif tecla == "c":
            self.limpar()
            return
        elif tecla == "escape":
            self.modo = "obstaculo"

        self.atualizar_status()

    def desenhar(self):
        self.canvas.delete("all")

        for linha in range(LINHAS):
            for coluna in range(COLUNAS):
                x1 = coluna * TAMANHO_CELULA
                y1 = linha * TAMANHO_CELULA
                x2 = x1 + TAMANHO_CELULA
                y2 = y1 + TAMANHO_CELULA

                cor = COR_LIVRE

                if self.mapa[linha][coluna] == 1:
                    cor = COR_OBSTACULO
                elif (linha, coluna) in self.rota and (linha, coluna) != self.inicio and (linha, coluna) != self.fim:
                    cor = COR_ROTA

                self.canvas.create_rectangle(x1, y1, x2, y2, fill=cor, outline=COR_GRID)

        if self.inicio is not None:
            linha, coluna = self.inicio
            x1 = coluna * TAMANHO_CELULA
            y1 = linha * TAMANHO_CELULA
            x2 = x1 + TAMANHO_CELULA
            y2 = y1 + TAMANHO_CELULA
            self.canvas.create_rectangle(x1, y1, x2, y2, fill=COR_INICIO, outline=COR_GRID)
            self.canvas.create_text((x1 + x2)//2, (y1 + y2)//2, text="I", fill="white")

        if self.fim is not None:
            linha, coluna = self.fim
            x1 = coluna * TAMANHO_CELULA
            y1 = linha * TAMANHO_CELULA
            x2 = x1 + TAMANHO_CELULA
            y2 = y1 + TAMANHO_CELULA
            self.canvas.create_rectangle(x1, y1, x2, y2, fill=COR_FIM, outline=COR_GRID)
            self.canvas.create_text((x1 + x2)//2, (y1 + y2)//2, text="F", fill="white")


if __name__ == "__main__":
    root = tk.Tk()
    app = EditorMapa(root)
    root.mainloop()