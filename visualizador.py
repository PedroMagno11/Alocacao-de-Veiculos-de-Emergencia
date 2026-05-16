import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D


class VisualizadorGRASP:
    """
    Gráfico ao vivo com 4 painéis que atualizam a cada iteração do GRASP:
      - Score Total
      - Score Ambulâncias A
      - Score Ambulâncias B
      - Regiões Cobertas

    Uso:
        viz = VisualizadorGRASP(total_iteracoes=100)
        viz.atualizar(iteracao=1, avaliacao=avaliacao, eh_melhor_global=True)
        viz.finalizar()
    """

    def __init__(self, total_iteracoes: int):
        self.total_iteracoes = total_iteracoes

        self.historico = {
            "score_total":                    [],
            "score_ambulancia_tipo_A":        [],
            "score_ambulancia_tipo_B":        [],
            "quant_regioes_cobertas_total":   [],
        }

        self.melhor_historico = {
            "score_total":                    [],
            "score_ambulancia_tipo_A":        [],
            "score_ambulancia_tipo_B":        [],
            "quant_regioes_cobertas_total":   [],
        }

        self._melhor_atual = {
            "score_total":                    0.0,
            "score_ambulancia_tipo_A":        0.0,
            "score_ambulancia_tipo_B":        0.0,
            "quant_regioes_cobertas_total":   0,
        }

        self._setup_figura()

    def _setup_figura(self):
        plt.ion()
        self.fig = plt.figure(figsize=(14, 8))
        self.fig.suptitle("Evolução do GRASP", fontsize=14, fontweight="bold")

        gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

        configs = [
            (gs[0, 0], "score_total",                   "Score Total",           "#2196F3"),
            (gs[0, 1], "score_ambulancia_tipo_A",        "Score — Ambulâncias A", "#4CAF50"),
            (gs[1, 0], "score_ambulancia_tipo_B",        "Score — Ambulâncias B", "#FF9800"),
            (gs[1, 1], "quant_regioes_cobertas_total",   "Regiões Cobertas",      "#9C27B0"),
        ]

        self.axes          = {}
        self.linhas        = {}
        self.melhor_linhas = {}

        for spec, chave, titulo, cor in configs:
            ax = self.fig.add_subplot(spec)
            ax.set_title(titulo, fontsize=10, fontweight="bold")
            ax.set_xlabel("Iteração", fontsize=8)
            ax.set_xlim(1, self.total_iteracoes)
            ax.tick_params(labelsize=8)
            ax.grid(True, linestyle="--", alpha=0.4)

            linha_iter,   = ax.plot([], [], color=cor,   linewidth=1.2, alpha=0.7, label="Iteração")
            linha_melhor, = ax.plot([], [], color="red", linewidth=1.5, linestyle="--", label="Melhor global")

            self.axes[chave]          = ax
            self.linhas[chave]        = linha_iter
            self.melhor_linhas[chave] = linha_melhor

        legenda_elementos = [
            Line2D([0], [0], color="gray", linewidth=1.2, alpha=0.7, label="Iteração atual"),
            Line2D([0], [0], color="red",  linewidth=1.5, linestyle="--", label="Melhor global"),
        ]
        self.fig.legend(handles=legenda_elementos, loc="lower center",
                        ncol=2, fontsize=9, frameon=True)

        plt.show()

    def atualizar(self, iteracao: int, avaliacao: dict, eh_melhor_global: bool):
        for chave in self.historico:
            valor = avaliacao[chave]
            self.historico[chave].append(valor)

            if eh_melhor_global:
                self._melhor_atual[chave] = valor

            self.melhor_historico[chave].append(self._melhor_atual[chave])

        xs = list(range(1, iteracao + 1))

        for chave, ax in self.axes.items():
            self.linhas[chave].set_data(xs, self.historico[chave])
            self.melhor_linhas[chave].set_data(xs, self.melhor_historico[chave])

            todos = self.historico[chave] + self.melhor_historico[chave]
            if todos:
                margem = (max(todos) - min(todos)) * 0.1 or 1
                ax.set_ylim(min(todos) - margem, max(todos) + margem)

        self.fig.suptitle(
            f"Evolução do GRASP — Iteração {iteracao}/{self.total_iteracoes}",
            fontsize=14, fontweight="bold"
        )

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def finalizar(self):
        self.fig.suptitle(
            f"Evolução do GRASP — Concluído ({self.total_iteracoes} iterações)",
            fontsize=14, fontweight="bold"
        )
        self.fig.canvas.draw()
        plt.ioff()
        plt.show()
