"""
experimentos_artigo.py
======================
Script de experimentos para o artigo DM-GRASP / PMCS-FA.

Gera:
  - resultados_artigo.csv  : tabela comparativa (Exato x GRASP x DM-GRASP)
  - fig1_convergencia.png  : evolução da FO por iteração (Fase 1 vs Fase 2)
  - fig2_antes_apos_bl.png : FO antes e após busca local (Fase 1 + Fase 2)
  - fig3_boxplot.png       : boxplot das FOs por método e tamanho de instância

Pré-requisitos
--------------
  pip install gurobipy pandas matplotlib numpy
  # (gurobipy só é necessário se quiser rodar o modelo exato)

Como usar
---------
  # Passo 1 — gere as instâncias de teste (se ainda não existirem)
  python instances/gerar_instancias.py --quantidades 50 100 200 500

  # Passo 2 — rode os experimentos
  python experimentos_artigo.py

  # Para pular o modelo exato (sem licença Gurobi):
  python experimentos_artigo.py --sem-exato

  # Para rodar só uma instância específica (debug rápido):
  python experimentos_artigo.py --instancias 50

  # Para definir tempo limite de 30 minutos por execução:
  python experimentos_artigo.py --tempo-limite 1800

  # Combinando flags:
  python experimentos_artigo.py --sem-exato --tempo-limite 3600 --instancias 100 200

Saídas
------
  Todos os arquivos são gravados em results/
"""

import argparse
import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # sem display — salva direto em arquivo
import matplotlib.pyplot as plt
import numpy as np

# ──────────────────────────────────────────────────────────────
# Parâmetros experimentais
# ──────────────────────────────────────────────────────────────

SEMENTES          = [42, 43, 44, 45, 46]   # 5 repetições independentes
TAMANHOS          = [50, 100, 200, 500]     # pontos de demanda por instância
PASTA_INSTANCIAS  = Path("instances")
PASTA_RESULTADOS  = Path("results")

# Parâmetros dos métodos heurísticos
ALPHA             = 0.6
ITERACOES_FASE1   = 100
ITERACOES_FASE2   = 200
MAX_ITER_SEM_MEL  = None
TAM_ELITE         = 40
FREQ_MINIMA       = 0.3

# Limite de tempo por execução (segundos). None = sem limite.
# Exemplos: 3600 = 1 hora | 1800 = 30 min | None = só por iterações
TEMPO_LIMITE_S        = 3600
PROPORCAO_TEMPO_FASE1 = 0.4   # 40% Fase 1, 60% Fase 2


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _nome_instancia(tamanho: int, indice: int = 1) -> str:
    return PASTA_INSTANCIAS / f"instancia_aleatoria_{indice:02d}_{tamanho}p.csv"


def _percentual_cobertura(solucao, pontos_cobertos, total_pontos: int) -> float:
    cobertos = set()
    for (regiao, tipo) in solucao:
        cobertos |= pontos_cobertos[regiao][tipo]
    return 100.0 * len(cobertos) / total_pontos


# ──────────────────────────────────────────────────────────────
# Modelo Exato (só instâncias pequenas)
# ──────────────────────────────────────────────────────────────

def rodar_exato(dataframe, tipos_ambulancia, qtd_maxima):
    """Retorna (fo_otimo, tempo_s) ou (None, None) se Gurobi não disponível."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
        from modelo_exato.modelo_exato import construir_modelo_pmcs_fa
        from instances.read_instance import pre_computar_pontos_cobertos

        regioes = [int(i) for i in dataframe["local_id"].tolist()]
        pontos_cobertos = pre_computar_pontos_cobertos(dataframe, tipos_ambulancia)

        modelo, x, _ = construir_modelo_pmcs_fa(
            regioes=regioes,
            tipos_ambulancia=tipos_ambulancia,
            quantidade_maxima_por_tipo=qtd_maxima,
            pontos_cobertos=pontos_cobertos,
        )
        modelo.setParam("OutputFlag", 0)   # silencia o Gurobi

        t0 = time.perf_counter()
        modelo.optimize()
        tempo = time.perf_counter() - t0

        if modelo.Status == GRB.OPTIMAL:
            return modelo.ObjVal, tempo
        return None, None
    except Exception as e:
        print(f"    [Exato] Falhou: {e}")
        return None, None


# ──────────────────────────────────────────────────────────────
# GRASP
# ──────────────────────────────────────────────────────────────

def rodar_grasp(dataframe, tipos_ambulancia, qtd_maxima, semente, tempo_limite_s=None):
    from grasp.grasp import executar_grasp

    t0 = time.perf_counter()
    resultado = executar_grasp(
        dataframe=dataframe,
        tipos_ambulancia=tipos_ambulancia,
        quantidade_maxima_por_tipo=qtd_maxima,
        parametro_alpha=ALPHA,
        max_iteracoes=ITERACOES_FASE1 + ITERACOES_FASE2,   # mesmo orçamento total
        max_iteracoes_sem_melhora=MAX_ITER_SEM_MEL,
        semente_aleatoria=semente,
        tempo_limite_s=tempo_limite_s,
    )
    tempo = time.perf_counter() - t0

    melhor_solucao, melhor_fo, pontos_cobertos, historico_fo, hist_antes, hist_pos = resultado
    return melhor_fo, tempo, historico_fo, hist_antes, hist_pos, pontos_cobertos


# ──────────────────────────────────────────────────────────────
# DM-GRASP
# ──────────────────────────────────────────────────────────────

def rodar_dm_grasp(dataframe, tipos_ambulancia, qtd_maxima, semente, tempo_limite_s=None):
    from dm_grasp.dm_grasp import executar_dm_grasp

    t0 = time.perf_counter()
    resultado = executar_dm_grasp(
        dataframe=dataframe,
        tipos_ambulancia=tipos_ambulancia,
        quantidade_maxima_por_tipo=qtd_maxima,
        parametro_alpha=ALPHA,
        iteracoes_fase1=ITERACOES_FASE1,
        iteracoes_fase2=ITERACOES_FASE2,
        max_iteracoes_sem_melhora=MAX_ITER_SEM_MEL,
        semente_aleatoria=semente,
        tamanho_memoria_elite=TAM_ELITE,
        frequencia_minima=FREQ_MINIMA,
        tempo_limite_s=tempo_limite_s,
        proporcao_tempo_fase1=PROPORCAO_TEMPO_FASE1,
    )
    tempo = time.perf_counter() - t0

    (
        melhor_solucao, melhor_fo, pontos_cobertos,
        hist_fo_f1, hist_fo_f2,
        hist_antes_f1, hist_apos_f1,
        hist_antes_f2, hist_apos_f2,
    ) = resultado

    return (
        melhor_fo, tempo,
        hist_fo_f1, hist_fo_f2,
        hist_antes_f1, hist_apos_f1,
        hist_antes_f2, hist_apos_f2,
        pontos_cobertos,
    )


# ──────────────────────────────────────────────────────────────
# Figuras
# ──────────────────────────────────────────────────────────────

def salvar_fig1_convergencia(hist_f1, hist_f2, tamanho, pasta):
    """
    Figura 1: evolucao do melhor acumulado da FO por iteracao.
    Painel unico mostrando Fase 1 (GRASP puro) e Fase 2 (DM-GRASP).
    """
    fig, ax = plt.subplots(figsize=(11, 5))

    n1    = len(hist_f1)
    n2    = len(hist_f2)
    todos = hist_f1 + hist_f2

    iter_f1 = list(range(1, n1 + 1))
    iter_f2 = list(range(n1 + 1, n1 + n2 + 1))

    ax.scatter(iter_f1, hist_f1, s=22, alpha=0.65,
               color="#4C72B0", zorder=3, label="GRASP puro (Fase 1)")
    ax.scatter(iter_f2, hist_f2, s=22, alpha=0.65,
               color="#DD8452", zorder=3, label="DM-GRASP (Fase 2)")

    melhor_acum = np.maximum.accumulate(todos)
    ax.plot(range(1, len(todos) + 1), melhor_acum,
            color="#222222", linewidth=1.8, linestyle="--",
            zorder=4, label="Melhor acumulado")

    ax.axvline(x=n1 + 0.5, color="#888888", linestyle=":", linewidth=1.2)
    ymin = ax.get_ylim()[0]
    ax.text(n1 + 1.5, ymin + 0.3, "Fase 2 ->",
            fontsize=8, color="#888888", va="bottom")

    ax.set_title(f"Evolucao da Funcao Objetivo -- {tamanho} pontos",
                 fontsize=13, pad=10)
    ax.set_xlabel("Iteracao", fontsize=11)
    ax.set_ylabel("Funcao Objetivo", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.25, linestyle="--")
    fig.tight_layout()

    caminho = pasta / f"fig1_convergencia_{tamanho}p.png"
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {caminho}")


def salvar_fig2_antes_apos_bl(
    hist_antes_grasp, hist_apos_grasp,
    hist_antes_dm,    hist_apos_dm,
    tamanho, pasta,
):
    """
    Figura 2: dois subplots lado a lado.
      Esquerda  -- GRASP puro:
          azul claro  (#6baed6) = pre busca local
          roxo/indigo (#4a1486) = pos busca local
      Direita -- DM-GRASP:
          laranja claro (#fdae6b) = pre busca local
          vermelho      (#cb181d) = pos busca local
    """
    fig, (ax_g, ax_dm) = plt.subplots(
        1, 2,
        figsize=(14, 5),
        sharey=True,
        layout="constrained",
        gridspec_kw={"wspace": 0.08},
    )

    n_g  = len(hist_antes_grasp)
    n_dm = len(hist_antes_dm)

    iter_g  = range(1, n_g  + 1)
    iter_dm = range(1, n_dm + 1)

    # GRASP puro (esquerda)
    ax_g.scatter(iter_g, hist_antes_grasp,
                 s=22, alpha=0.45, color="#6baed6", zorder=2,
                 label="Pre-BL")
    ax_g.scatter(iter_g, hist_apos_grasp,
                 s=22, alpha=0.85, color="#4a1486", zorder=3,
                 label="Pos-BL")
    melhor_g = np.maximum.accumulate(hist_apos_grasp)
    ax_g.plot(iter_g, melhor_g,
              color="#222222", linewidth=1.6, linestyle="--",
              zorder=4, label="Melhor acumulado")
    ax_g.set_title("GRASP puro", fontsize=12, pad=8)
    ax_g.set_xlabel("Iteracao", fontsize=11)
    ax_g.set_ylabel("Funcao Objetivo", fontsize=11)
    ax_g.legend(fontsize=9, framealpha=0.9)
    ax_g.grid(True, alpha=0.25, linestyle="--")

    # DM-GRASP (direita)
    ax_dm.scatter(iter_dm, hist_antes_dm,
                  s=22, alpha=0.45, color="#fdae6b", zorder=2,
                  label="Pre-BL")
    ax_dm.scatter(iter_dm, hist_apos_dm,
                  s=22, alpha=0.85, color="#cb181d", zorder=3,
                  label="Pos-BL")
    melhor_dm = np.maximum.accumulate(hist_apos_dm)
    ax_dm.plot(iter_dm, melhor_dm,
               color="#222222", linewidth=1.6, linestyle="--",
               zorder=4, label="Melhor acumulado")
    ax_dm.set_title("DM-GRASP", fontsize=12, pad=8)
    ax_dm.set_xlabel("Iteracao", fontsize=11)
    ax_dm.legend(fontsize=9, framealpha=0.9)
    ax_dm.grid(True, alpha=0.25, linestyle="--")

    fig.suptitle(
        f"FO antes e apos Busca Local -- {tamanho} pontos",
        fontsize=13,
    )

    caminho = pasta / f"fig2_antes_apos_bl_{tamanho}p.png"
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {caminho}")


def salvar_fig3_boxplot(dados_boxplot, pasta):
    """
    Figura 3: boxplot comparando GRASP x DM-GRASP por tamanho de instância.

    dados_boxplot = {
        tamanho: {
            "GRASP":    [fo1, fo2, ...],
            "DM-GRASP": [fo1, fo2, ...],
        }, ...
    }
    """
    tamanhos_ord = sorted(dados_boxplot.keys())
    n = len(tamanhos_ord)

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, tam in zip(axes, tamanhos_ord):
        grupos = dados_boxplot[tam]
        rotulos = list(grupos.keys())
        valores = [grupos[r] for r in rotulos]

        bp = ax.boxplot(valores, tick_labels=rotulos, patch_artist=True)
        cores = ["steelblue", "darkorange"]
        for patch, cor in zip(bp["boxes"], cores):
            patch.set_facecolor(cor)
            patch.set_alpha(0.7)

        ax.set_title(f"{tam} pontos")
        ax.set_ylabel("Função Objetivo")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Distribuição das Soluções — GRASP vs DM-GRASP", fontsize=12)
    fig.tight_layout()

    caminho = pasta / "fig3_boxplot.png"
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    print(f"  Salvo: {caminho}")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main(sem_exato=False, tamanhos_filtro=None):
    PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)

    from instances.read_instance import ler_instancia
    import config.PARAMETROS as P

    tamanhos = tamanhos_filtro if tamanhos_filtro else TAMANHOS

    # ── Tabela de resultados ──────────────────────────────────
    cabecalho = [
        "instancia", "n_pontos", "semente",
        "exato_fo", "exato_tempo_s",
        "grasp_fo", "grasp_tempo_s",
        "dmgrasp_fo", "dmgrasp_tempo_s",
        "gap_grasp_pct", "gap_dmgrasp_pct",
    ]

    linhas_csv = []
    dados_boxplot = {}   # para fig3

    # Guarda históricos do DM-GRASP de uma instância representativa
    # (200p ou o maior disponível) para as figuras 1 e 2
    historicos_fig = {}

    for tam in tamanhos:
        dados_boxplot[tam] = {"GRASP": [], "DM-GRASP": []}
        caminho = _nome_instancia(tam)

        if not caminho.exists():
            print(f"\n[AVISO] Instância não encontrada: {caminho}")
            print(f"         Execute: python instances/gerar_instancias.py --quantidades {tam}")
            continue

        dataframe = ler_instancia(str(caminho))
        total_pontos = len(dataframe)
        print(f"\n{'=' * 60}")
        print(f"  Instância: {caminho.name}  ({total_pontos} pontos)")
        print(f"{'=' * 60}")

        # Modelo exato (somente instâncias pequenas e sem flag --sem-exato)
        fo_exato = tempo_exato = None
        if not sem_exato and tam <= 50:
            print("  [Exato] Rodando Gurobi...", flush=True)
            fo_exato, tempo_exato = rodar_exato(dataframe, P.TIPOS_AMBULANCIA, P.QUANTIDADE_MAXIMA_POR_TIPO)
            if fo_exato is not None:
                print(f"  [Exato] FO = {fo_exato:.2f}  Tempo = {tempo_exato:.2f}s")

        fos_grasp    = []
        fos_dmgrasp  = []

        for semente in SEMENTES:
            print(f"  Semente {semente}...", end=" ", flush=True)

            # GRASP
            fo_g, t_g, hist_fo_g, hist_antes_g, hist_apos_g, _ = rodar_grasp(
                dataframe, P.TIPOS_AMBULANCIA, P.QUANTIDADE_MAXIMA_POR_TIPO,
                semente, tempo_limite_s=TEMPO_LIMITE_S,
            )
            fos_grasp.append(fo_g)

            # DM-GRASP
            (
                fo_dm, t_dm,
                hist_fo_f1, hist_fo_f2,
                hist_antes_f1, hist_apos_f1,
                hist_antes_f2, hist_apos_f2,
                _pontos_cobertos,
            ) = rodar_dm_grasp(
                dataframe, P.TIPOS_AMBULANCIA, P.QUANTIDADE_MAXIMA_POR_TIPO,
                semente, tempo_limite_s=TEMPO_LIMITE_S,
            )
            fos_dmgrasp.append(fo_dm)

            # Guarda histórico da primeira semente para as figuras
            if semente == SEMENTES[0]:
                historicos_fig[tam] = {
                    # GRASP puro
                    "hist_antes_g":  hist_antes_g,
                    "hist_apos_g":   hist_apos_g,
                    # DM-GRASP
                    "hist_fo_f1":    hist_fo_f1,
                    "hist_fo_f2":    hist_fo_f2,
                    "hist_antes_dm": hist_antes_f1 + hist_antes_f2,
                    "hist_apos_dm":  hist_apos_f1  + hist_apos_f2,
                }

            # Gap em relação ao exato (se disponível) ou ao DM-GRASP
            referencia = fo_exato if fo_exato else fo_dm
            gap_g  = 100.0 * (referencia - fo_g)  / referencia if referencia else None
            gap_dm = 100.0 * (referencia - fo_dm) / referencia if (referencia and fo_exato) else None

            linhas_csv.append({
                "instancia":       caminho.name,
                "n_pontos":        total_pontos,
                "semente":         semente,
                "exato_fo":        f"{fo_exato:.4f}" if fo_exato else "",
                "exato_tempo_s":   f"{tempo_exato:.4f}" if tempo_exato else "",
                "grasp_fo":        f"{fo_g:.4f}",
                "grasp_tempo_s":   f"{t_g:.4f}",
                "dmgrasp_fo":      f"{fo_dm:.4f}",
                "dmgrasp_tempo_s": f"{t_dm:.4f}",
                "gap_grasp_pct":   f"{gap_g:.2f}" if gap_g is not None else "",
                "gap_dmgrasp_pct": f"{gap_dm:.2f}" if gap_dm is not None else "",
            })

            print(f"GRASP={fo_g:.1f}  DM-GRASP={fo_dm:.1f}")

        dados_boxplot[tam]["GRASP"]    = fos_grasp
        dados_boxplot[tam]["DM-GRASP"] = fos_dmgrasp

        # ── Resumo por tamanho ────────────────────────────────
        print(f"\n  Resumo {tam}p:")
        print(f"    GRASP    : média={np.mean(fos_grasp):.2f}  dp={np.std(fos_grasp):.2f}  melhor={max(fos_grasp):.2f}")
        print(f"    DM-GRASP : média={np.mean(fos_dmgrasp):.2f}  dp={np.std(fos_dmgrasp):.2f}  melhor={max(fos_dmgrasp):.2f}")
        if fo_exato:
            gap_medio_g  = 100.0 * (fo_exato - np.mean(fos_grasp))  / fo_exato
            gap_medio_dm = 100.0 * (fo_exato - np.mean(fos_dmgrasp)) / fo_exato
            print(f"    Exato    : {fo_exato:.2f}")
            print(f"    Gap GRASP (médio)    : {gap_medio_g:.2f}%")
            print(f"    Gap DM-GRASP (médio) : {gap_medio_dm:.2f}%")

        # ── Figuras 1 e 2 para este tamanho ──────────────────
        if tam in historicos_fig:
            h = historicos_fig[tam]
            salvar_fig1_convergencia(h["hist_fo_f1"], h["hist_fo_f2"], tam, PASTA_RESULTADOS)
            salvar_fig2_antes_apos_bl(
                h["hist_antes_g"],  h["hist_apos_g"],
                h["hist_antes_dm"], h["hist_apos_dm"],
                tam, PASTA_RESULTADOS,
            )

    # ── CSV completo ──────────────────────────────────────────
    caminho_csv = PASTA_RESULTADOS / "resultados_artigo.csv"
    with open(caminho_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cabecalho)
        writer.writeheader()
        writer.writerows(linhas_csv)
    print(f"\nTabela completa salva em: {caminho_csv}")

    # ── Figura 3: boxplot ─────────────────────────────────────
    if dados_boxplot:
        salvar_fig3_boxplot(dados_boxplot, PASTA_RESULTADOS)

    # ── Tabela resumo para o artigo (média ± dp) ──────────────
    print("\n" + "=" * 70)
    print("  TABELA RESUMO PARA O ARTIGO")
    print("=" * 70)
    fmt = "{:<12} {:>12} {:>12} {:>22} {:>22} {:>14} {:>14}"
    print(fmt.format("Instância", "Exato (FO)", "T.Exato(s)",
                     "GRASP (FO ± dp)", "DM-GRASP (FO ± dp)",
                     "Gap GRASP(%)", "Gap DM(%)"))
    print("-" * 110)

    for tam in tamanhos:
        fos_g  = [float(l["grasp_fo"])   for l in linhas_csv if l["n_pontos"] == tam and l["grasp_fo"]]
        fos_dm = [float(l["dmgrasp_fo"]) for l in linhas_csv if l["n_pontos"] == tam and l["dmgrasp_fo"]]
        exato_vals = [float(l["exato_fo"]) for l in linhas_csv if l["n_pontos"] == tam and l["exato_fo"]]
        exato_str  = f"{exato_vals[0]:.2f}" if exato_vals else "—"
        tempo_ex   = [float(l["exato_tempo_s"]) for l in linhas_csv if l["n_pontos"] == tam and l["exato_tempo_s"]]
        tempo_ex_str = f"{tempo_ex[0]:.1f}" if tempo_ex else "—"

        ref = exato_vals[0] if exato_vals else (max(fos_dm) if fos_dm else None)
        gap_g  = f"{100.0*(ref-np.mean(fos_g))/ref:.2f}" if ref and fos_g else "—"
        gap_dm = f"{100.0*(ref-np.mean(fos_dm))/ref:.2f}" if (ref and exato_vals and fos_dm) else "—"

        print(fmt.format(
            f"{tam}p",
            exato_str, tempo_ex_str,
            f"{np.mean(fos_g):.2f} ± {np.std(fos_g):.2f}" if fos_g else "—",
            f"{np.mean(fos_dm):.2f} ± {np.std(fos_dm):.2f}" if fos_dm else "—",
            gap_g, gap_dm,
        ))

    print("=" * 70)
    print("\nExperimentos concluídos. Arquivos em:", PASTA_RESULTADOS)


# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sem-exato", action="store_true",
        help="Pula o modelo exato (útil sem licença Gurobi)"
    )
    parser.add_argument(
        "--instancias", nargs="+", type=int, default=None,
        help="Roda apenas os tamanhos informados (ex: --instancias 50 100)"
    )
    parser.add_argument(
        "--tempo-limite", type=int, default=None,
        help="Tempo limite por execução em segundos (ex: 3600 = 1h). "
             "Sobrescreve TEMPO_LIMITE_S definido no script."
    )
    args = parser.parse_args()

    if args.tempo_limite is not None:
        TEMPO_LIMITE_S = args.tempo_limite

    main(sem_exato=args.sem_exato, tamanhos_filtro=args.instancias)