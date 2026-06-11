"""
experimentos_artigo.py
======================
Script de experimentos para o artigo DM-GRASP / PMCS-FA.

Este script percorre automaticamente as instâncias em:

    instances/100p/3_clusters/desvio_2_0km/*.csv

E salva os resultados espelhando a estrutura em:

    results/100p/3_clusters/desvio_2_0km/<nome_da_instancia>/

Além disso, gera consolidados por cenário e um consolidado geral.
"""

import argparse
import csv
import re
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # sem display — salva direto em arquivo
import matplotlib.pyplot as plt
import numpy as np

# ──────────────────────────────────────────────────────────────
# Parâmetros experimentais
# ──────────────────────────────────────────────────────────────
import config.PARAMETROS as P

ALPHA = P.PARAMETRO_ALPHA
MAX_ITER_SEM_MEL = P.MAX_ITERACOES_SEM_MELHORA
SEMENTES = [P.SEMENTE_ALEATORIA]

TAMANHOS          = [100, 500, 1000, 1500, 2000]
PASTA_INSTANCIAS  = Path("instances")
PASTA_RESULTADOS  = Path("results")

ITERACOES_FASE1   = 150
ITERACOES_FASE2   = 150
TAM_ELITE         = 40
FREQ_MINIMA       = 0.3

# Limite de tempo por execução (segundos). None = sem limite.
TEMPO_LIMITE_S        = 3600
PROPORCAO_TEMPO_FASE1 = 0.4


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class InstanciaExperimento:
    caminho: Path
    tamanho: int
    clusters: int | None
    desvio_cluster_km: float | None

    @property
    def id_experimento(self) -> str:
        partes = [f"{self.tamanho}p"]
        if self.clusters is not None:
            partes.append(f"{self.clusters}c")
        if self.desvio_cluster_km is not None:
            partes.append(f"desvio_{_formatar_desvio(self.desvio_cluster_km)}km")
        return "_".join(partes)

    @property
    def nome_instancia(self) -> str:
        return self.caminho.stem


def _formatar_desvio(valor: float) -> str:
    return str(valor).replace(".", "_")


def _extrair_primeiro_inteiro_padrao(texto: str, padrao: str) -> int | None:
    match = re.search(padrao, texto, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _extrair_desvio(texto: str) -> float | None:
    match = re.search(r"desvio[_-]?(\d+(?:[._]\d+)?)\s*km?", texto, flags=re.IGNORECASE)
    if not match:
        return None
    return float(match.group(1).replace("_", "."))


def _descobrir_instancias(
    pasta_instancias: Path,
    tamanhos_filtro: list[int] | None = None,
    clusters_filtro: list[int] | None = None,
    desvios_filtro: list[float] | None = None,
) -> list[InstanciaExperimento]:
    """
    Descobre instâncias no formato novo:

        instances/100p/3_clusters/desvio_2_0km/*.csv

    Também aceita o formato antigo:

        instances/instancia_aleatoria_01_100p.csv
    """
    tamanhos_set = set(tamanhos_filtro) if tamanhos_filtro else None
    clusters_set = set(clusters_filtro) if clusters_filtro else None
    desvios_set = {round(v, 6) for v in desvios_filtro} if desvios_filtro else None

    instancias = []
    for caminho in sorted(pasta_instancias.rglob("*.csv")):
        relativo = caminho.relative_to(pasta_instancias)
        partes = relativo.parts
        texto_relativo = str(relativo)

        tamanho = None
        clusters = None
        desvio = None

        # Preferência: metadados nas pastas do novo formato.
        for parte in partes:
            tamanho = tamanho or _extrair_primeiro_inteiro_padrao(parte, r"(\d+)p")
            clusters = clusters or _extrair_primeiro_inteiro_padrao(parte, r"(\d+)[_-]?clusters?")
            desvio = desvio if desvio is not None else _extrair_desvio(parte)

        # Fallback: metadados no nome do arquivo/formato antigo.
        tamanho = tamanho or _extrair_primeiro_inteiro_padrao(caminho.name, r"_(\d+)p\.csv$")
        tamanho = tamanho or _extrair_primeiro_inteiro_padrao(texto_relativo, r"(\d+)p")
        clusters = clusters or _extrair_primeiro_inteiro_padrao(texto_relativo, r"(\d+)c(?:_|\b)")
        desvio = desvio if desvio is not None else _extrair_desvio(texto_relativo)

        if tamanho is None:
            print(f"[AVISO] Ignorando CSV sem quantidade de pontos identificável: {caminho}")
            continue
        if tamanhos_set and tamanho not in tamanhos_set:
            continue
        if clusters_set and clusters not in clusters_set:
            continue
        if desvios_set and (desvio is None or round(desvio, 6) not in desvios_set):
            continue

        instancias.append(InstanciaExperimento(caminho, tamanho, clusters, desvio))

    return instancias


def _obter_pasta_cenario_resultado(instancia: InstanciaExperimento) -> Path:
    """
    Espelha a pasta da instância dentro de results/.

    Exemplo:
        instances/100p/3_clusters/desvio_2_0km/instancia_01.csv
    vira:
        results/100p/3_clusters/desvio_2_0km/
    """
    try:
        relativo_pai = instancia.caminho.parent.relative_to(PASTA_INSTANCIAS)
    except ValueError:
        relativo_pai = Path(instancia.id_experimento)

    # Formato antigo: instances/instancia_aleatoria_01_100p.csv
    if str(relativo_pai) == ".":
        relativo_pai = Path("formato_antigo") / instancia.id_experimento

    pasta = PASTA_RESULTADOS / relativo_pai
    pasta.mkdir(parents=True, exist_ok=True)
    return pasta


def _obter_pasta_instancia_resultado(instancia: InstanciaExperimento) -> Path:
    """
    Cria uma pasta específica para cada arquivo de instância.

    Exemplo:
        results/100p/3_clusters/desvio_2_0km/instancia_aleatoria_01_100p/
    """
    pasta = _obter_pasta_cenario_resultado(instancia) / instancia.nome_instancia
    pasta.mkdir(parents=True, exist_ok=True)
    return pasta


def _obter_pasta_consolidado_cenario(instancia: InstanciaExperimento) -> Path:
    pasta = _obter_pasta_cenario_resultado(instancia) / "consolidado"
    pasta.mkdir(parents=True, exist_ok=True)
    return pasta


def _escrever_csv(caminho: Path, cabecalho: list[str], linhas: list[dict]) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cabecalho)
        writer.writeheader()
        writer.writerows(linhas)
    print(f"  Salvo: {caminho}")


def _percentual_cobertura(solucao, pontos_cobertos, total_pontos: int) -> float:
    cobertos = set()
    for (regiao, tipo) in solucao:
        cobertos |= pontos_cobertos[regiao][tipo]
    return 100.0 * len(cobertos) / total_pontos


# ──────────────────────────────────────────────────────────────
# Modelo Exato
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
        modelo.setParam("OutputFlag", 1)
        modelo.setParam("SoftMemLimit", 8)
        modelo.setParam("TimeLimit", TEMPO_LIMITE_S)

        t0 = time.perf_counter()
        modelo.optimize()
        tempo = time.perf_counter() - t0

        if modelo.SolCount > 0:
            print(
                f"    [Exato] Status={modelo.Status} "
                f"FO={modelo.ObjVal:.2f} "
                f"Gap={modelo.MIPGap*100:.2f}%"
            )
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
        max_iteracoes=ITERACOES_FASE1 + ITERACOES_FASE2,
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

def salvar_fig1_convergencia(hist_f1, hist_f2, titulo, pasta):
    fig, ax = plt.subplots(figsize=(11, 5))

    n1 = len(hist_f1)
    n2 = len(hist_f2)
    todos = hist_f1 + hist_f2

    iter_f1 = list(range(1, n1 + 1))
    iter_f2 = list(range(n1 + 1, n1 + n2 + 1))

    ax.scatter(iter_f1, hist_f1, s=22, alpha=0.65,
               color="#4C72B0", zorder=3, label="GRASP puro (Fase 1)")
    ax.scatter(iter_f2, hist_f2, s=22, alpha=0.65,
               color="#DD8452", zorder=3, label="DM-GRASP (Fase 2)")

    if todos:
        melhor_acum = np.maximum.accumulate(todos)
        ax.plot(range(1, len(todos) + 1), melhor_acum,
                color="#222222", linewidth=1.8, linestyle="--",
                zorder=4, label="Melhor acumulado")

    ax.axvline(x=n1 + 0.5, color="#888888", linestyle=":", linewidth=1.2)
    ymin = ax.get_ylim()[0]
    ax.text(n1 + 1.5, ymin + 0.3, "Fase 2 ->",
            fontsize=8, color="#888888", va="bottom")

    ax.set_title(f"Evolução da Função Objetivo -- {titulo}", fontsize=13, pad=10)
    ax.set_xlabel("Iteração", fontsize=11)
    ax.set_ylabel("Função Objetivo", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.25, linestyle="--")
    fig.tight_layout()

    caminho = pasta / "fig1_convergencia.png"
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {caminho}")


def salvar_fig2_antes_apos_bl(
    hist_antes_grasp, hist_apos_grasp,
    hist_antes_dm, hist_apos_dm,
    titulo, pasta,
):
    fig, (ax_g, ax_dm) = plt.subplots(
        1, 2,
        figsize=(14, 5),
        sharey=True,
        layout="constrained",
        gridspec_kw={"wspace": 0.08},
    )

    n_g = len(hist_antes_grasp)
    n_dm = len(hist_antes_dm)

    iter_g = range(1, n_g + 1)
    iter_dm = range(1, n_dm + 1)

    ax_g.scatter(iter_g, hist_antes_grasp,
                 s=22, alpha=0.45, color="#6baed6", zorder=2,
                 label="Pré-BL")
    ax_g.scatter(iter_g, hist_apos_grasp,
                 s=22, alpha=0.85, color="#4a1486", zorder=3,
                 label="Pós-BL")
    if hist_apos_grasp:
        melhor_g = np.maximum.accumulate(hist_apos_grasp)
        ax_g.plot(iter_g, melhor_g,
                  color="#222222", linewidth=1.6, linestyle="--",
                  zorder=4, label="Melhor acumulado")
    ax_g.set_title("GRASP puro", fontsize=12, pad=8)
    ax_g.set_xlabel("Iteração", fontsize=11)
    ax_g.set_ylabel("Função Objetivo", fontsize=11)
    ax_g.legend(fontsize=9, framealpha=0.9)
    ax_g.grid(True, alpha=0.25, linestyle="--")

    ax_dm.scatter(iter_dm, hist_antes_dm,
                  s=22, alpha=0.45, color="#fdae6b", zorder=2,
                  label="Pré-BL")
    ax_dm.scatter(iter_dm, hist_apos_dm,
                  s=22, alpha=0.85, color="#cb181d", zorder=3,
                  label="Pós-BL")
    if hist_apos_dm:
        melhor_dm = np.maximum.accumulate(hist_apos_dm)
        ax_dm.plot(iter_dm, melhor_dm,
                   color="#222222", linewidth=1.6, linestyle="--",
                   zorder=4, label="Melhor acumulado")
    ax_dm.set_title("DM-GRASP", fontsize=12, pad=8)
    ax_dm.set_xlabel("Iteração", fontsize=11)
    ax_dm.legend(fontsize=9, framealpha=0.9)
    ax_dm.grid(True, alpha=0.25, linestyle="--")

    fig.suptitle(f"FO antes e após Busca Local -- {titulo}", fontsize=13)

    caminho = pasta / "fig2_antes_apos_bl.png"
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvo: {caminho}")


def salvar_fig3_boxplot(dados_boxplot, pasta, nome_arquivo="fig3_boxplot.png", titulo="Distribuição das Soluções — GRASP vs DM-GRASP"):
    """
    dados_boxplot = {
        "rotulo": {
            "GRASP": [fo1, fo2, ...],
            "DM-GRASP": [fo1, fo2, ...],
        }, ...
    }
    """
    chaves_ord = sorted(dados_boxplot.keys())
    n = len(chaves_ord)

    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(max(4 * n, 5), 5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, chave in zip(axes, chaves_ord):
        grupos = dados_boxplot[chave]
        rotulos = list(grupos.keys())
        valores = [grupos[r] for r in rotulos]

        bp = ax.boxplot(valores, tick_labels=rotulos, patch_artist=True)
        cores = ["steelblue", "darkorange"]
        for patch, cor in zip(bp["boxes"], cores):
            patch.set_facecolor(cor)
            patch.set_alpha(0.7)

        ax.set_title(str(chave), fontsize=9)
        ax.set_ylabel("Função Objetivo")
        ax.grid(True, alpha=0.3)

    fig.suptitle(titulo, fontsize=12)
    fig.tight_layout()

    caminho = pasta / nome_arquivo
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    print(f"  Salvo: {caminho}")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main(sem_exato=False, tamanhos_filtro=None, clusters_filtro=None, desvios_filtro=None):
    PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)

    from instances.read_instance import ler_instancia
    import config.PARAMETROS as P

    # Se o usuário não passou --instancias, percorre todos os CSVs encontrados.
    tamanhos = tamanhos_filtro

    instancias = _descobrir_instancias(
        PASTA_INSTANCIAS,
        tamanhos_filtro=tamanhos,
        clusters_filtro=clusters_filtro,
        desvios_filtro=desvios_filtro,
    )

    if not instancias:
        print("\n[AVISO] Nenhuma instância encontrada.")
        print("         Formato esperado: instances/100p/3_clusters/desvio_2_0km/*.csv")
        print("         Ou formato antigo: instances/instancia_aleatoria_01_100p.csv")
        return

    cabecalho = [
        "instancia", "arquivo_instancia", "caminho_instancia",
        "n_pontos", "clusters", "desvio_cluster_km", "semente",
        "exato_fo", "exato_tempo_s",
        "grasp_fo", "grasp_tempo_s",
        "dmgrasp_fo", "dmgrasp_tempo_s",
        "gap_grasp_pct", "gap_dmgrasp_pct",
    ]

    linhas_gerais = []
    linhas_por_cenario = {}
    boxplot_por_cenario = {}
    boxplot_geral = {}

    for instancia in instancias:
        tam = instancia.tamanho
        caminho = instancia.caminho
        pasta_instancia = _obter_pasta_instancia_resultado(instancia)
        pasta_consolidado_cenario = _obter_pasta_consolidado_cenario(instancia)
        chave_cenario = str(_obter_pasta_cenario_resultado(instancia).relative_to(PASTA_RESULTADOS))
        chave_instancia = f"{instancia.id_experimento}_{instancia.nome_instancia}"

        linhas_por_cenario.setdefault(chave_cenario, [])
        boxplot_por_cenario.setdefault(chave_cenario, {})

        dataframe = ler_instancia(str(caminho))
        total_pontos = len(dataframe)

        print(f"\n{'=' * 70}")
        print(f"  Cenário   : {chave_cenario}")
        print(f"  Instância : {instancia.nome_instancia}")
        print(f"  Arquivo   : {caminho}")
        print(f"  Pontos    : {total_pontos}")
        print(f"  Clusters  : {instancia.clusters if instancia.clusters is not None else '—'}")
        print(f"  Desvio    : {instancia.desvio_cluster_km if instancia.desvio_cluster_km is not None else '—'} km")
        print(f"  Saída     : {pasta_instancia}")
        print(f"{'=' * 70}")

        # Modelo exato somente para instâncias pequenas e sem --sem-exato.
        fo_exato = tempo_exato = None
        # if not sem_exato and tam <= 50:
        if not sem_exato:

            print("  [Exato] Rodando Gurobi...", flush=True)
            fo_exato, tempo_exato = rodar_exato(
                dataframe,
                P.TIPOS_AMBULANCIA,
                P.QUANTIDADE_MAXIMA_POR_TIPO,
            )
            if fo_exato is not None:
                print(f"  [Exato] FO = {fo_exato:.2f}  Tempo = {tempo_exato:.2f}s")

        fos_grasp = []
        fos_dmgrasp = []
        linhas_instancia = []
        historicos_fig = None

        for semente in SEMENTES:
            print(f"  Semente {semente}...", end=" ", flush=True)

            fo_g, t_g, hist_fo_g, hist_antes_g, hist_apos_g, _ = rodar_grasp(
                dataframe,
                P.TIPOS_AMBULANCIA,
                P.QUANTIDADE_MAXIMA_POR_TIPO,
                semente,
                tempo_limite_s=TEMPO_LIMITE_S,
            )
            fos_grasp.append(fo_g)

            (
                fo_dm, t_dm,
                hist_fo_f1, hist_fo_f2,
                hist_antes_f1, hist_apos_f1,
                hist_antes_f2, hist_apos_f2,
                _pontos_cobertos,
            ) = rodar_dm_grasp(
                dataframe,
                P.TIPOS_AMBULANCIA,
                P.QUANTIDADE_MAXIMA_POR_TIPO,
                semente,
                tempo_limite_s=TEMPO_LIMITE_S,
            )
            fos_dmgrasp.append(fo_dm)

            if historicos_fig is None:
                historicos_fig = {
                    "hist_antes_g": hist_antes_g,
                    "hist_apos_g": hist_apos_g,
                    "hist_fo_f1": hist_fo_f1,
                    "hist_fo_f2": hist_fo_f2,
                    "hist_antes_dm": hist_antes_f1 + hist_antes_f2,
                    "hist_apos_dm": hist_apos_f1 + hist_apos_f2,
                }

            # Gap científico: só preenche se houver ótimo/exato.
            if fo_exato:
                gap_g = 100.0 * (fo_exato - fo_g) / fo_exato
                gap_dm = 100.0 * (fo_exato - fo_dm) / fo_exato
            else:
                gap_g = None
                gap_dm = None

            linha = {
                "instancia": instancia.id_experimento,
                "arquivo_instancia": instancia.nome_instancia,
                "caminho_instancia": str(caminho),
                "n_pontos": total_pontos,
                "clusters": instancia.clusters if instancia.clusters is not None else "",
                "desvio_cluster_km": instancia.desvio_cluster_km if instancia.desvio_cluster_km is not None else "",
                "semente": semente,
                "exato_fo": f"{fo_exato:.4f}" if fo_exato else "",
                "exato_tempo_s": f"{tempo_exato:.4f}" if tempo_exato else "",
                "grasp_fo": f"{fo_g:.4f}",
                "grasp_tempo_s": f"{t_g:.4f}",
                "dmgrasp_fo": f"{fo_dm:.4f}",
                "dmgrasp_tempo_s": f"{t_dm:.4f}",
                "gap_grasp_pct": f"{gap_g:.2f}" if gap_g is not None else "",
                "gap_dmgrasp_pct": f"{gap_dm:.2f}" if gap_dm is not None else "",
            }

            linhas_instancia.append(linha)
            linhas_gerais.append(linha)
            linhas_por_cenario[chave_cenario].append(linha)

            print(f"GRASP={fo_g:.1f}  DM-GRASP={fo_dm:.1f}")

        print(f"\n  Resumo {chave_instancia}:")
        print(f"    GRASP    : média={np.mean(fos_grasp):.2f}  dp={np.std(fos_grasp):.2f}  melhor={max(fos_grasp):.2f}")
        print(f"    DM-GRASP : média={np.mean(fos_dmgrasp):.2f}  dp={np.std(fos_dmgrasp):.2f}  melhor={max(fos_dmgrasp):.2f}")
        if fo_exato:
            gap_medio_g = 100.0 * (fo_exato - np.mean(fos_grasp)) / fo_exato
            gap_medio_dm = 100.0 * (fo_exato - np.mean(fos_dmgrasp)) / fo_exato
            print(f"    Exato    : {fo_exato:.2f}")
            print(f"    Gap GRASP médio    : {gap_medio_g:.2f}%")
            print(f"    Gap DM-GRASP médio : {gap_medio_dm:.2f}%")
        else:
            print("    Exato    : não executado")
            print("    Gap      : não calculado sem ótimo/exato")

        # CSV e figuras específicos desta instância.
        _escrever_csv(pasta_instancia / "resultados.csv", cabecalho, linhas_instancia)

        if historicos_fig:
            titulo_fig = f"{instancia.id_experimento} / {instancia.nome_instancia}"
            salvar_fig1_convergencia(
                historicos_fig["hist_fo_f1"],
                historicos_fig["hist_fo_f2"],
                titulo_fig,
                pasta_instancia,
            )
            salvar_fig2_antes_apos_bl(
                historicos_fig["hist_antes_g"],
                historicos_fig["hist_apos_g"],
                historicos_fig["hist_antes_dm"],
                historicos_fig["hist_apos_dm"],
                titulo_fig,
                pasta_instancia,
            )

        dados_instancia = {
            instancia.nome_instancia: {
                "GRASP": fos_grasp,
                "DM-GRASP": fos_dmgrasp,
            }
        }
        salvar_fig3_boxplot(
            dados_instancia,
            pasta_instancia,
            nome_arquivo="fig3_boxplot.png",
            titulo=f"Distribuição das Soluções — {instancia.id_experimento} / {instancia.nome_instancia}",
        )

        # Dados para consolidados.
        boxplot_por_cenario[chave_cenario][instancia.nome_instancia] = {
            "GRASP": fos_grasp,
            "DM-GRASP": fos_dmgrasp,
        }
        boxplot_geral[chave_instancia] = {
            "GRASP": fos_grasp,
            "DM-GRASP": fos_dmgrasp,
        }

    # ── Consolidados por cenário ──────────────────────────────
    for chave_cenario, linhas in linhas_por_cenario.items():
        pasta_consolidado = PASTA_RESULTADOS / chave_cenario / "consolidado"
        pasta_consolidado.mkdir(parents=True, exist_ok=True)
        _escrever_csv(pasta_consolidado / "resultados_cenario.csv", cabecalho, linhas)

        salvar_fig3_boxplot(
            boxplot_por_cenario.get(chave_cenario, {}),
            pasta_consolidado,
            nome_arquivo="fig3_boxplot_cenario.png",
            titulo=f"Distribuição das Soluções — {chave_cenario}",
        )

    # ── Consolidado geral ─────────────────────────────────────
    pasta_consolidado_geral = PASTA_RESULTADOS / "consolidado"
    pasta_consolidado_geral.mkdir(parents=True, exist_ok=True)
    _escrever_csv(pasta_consolidado_geral / "resultados_artigo_geral.csv", cabecalho, linhas_gerais)

    salvar_fig3_boxplot(
        boxplot_geral,
        pasta_consolidado_geral,
        nome_arquivo="fig3_boxplot_geral.png",
        titulo="Distribuição das Soluções — Todos os Cenários",
    )

    # ── Tabela resumo para o artigo ───────────────────────────
    print("\n" + "=" * 90)
    print("  TABELA RESUMO PARA O ARTIGO")
    print("=" * 90)
    fmt = "{:<34} {:>12} {:>12} {:>22} {:>22} {:>14} {:>14}"
    print(fmt.format("Instância", "Exato (FO)", "T.Exato(s)",
                     "GRASP (FO ± dp)", "DM-GRASP (FO ± dp)",
                     "Gap GRASP(%)", "Gap DM(%)"))
    print("-" * 130)

    ids = sorted({(l["instancia"], l["arquivo_instancia"]) for l in linhas_gerais})
    for instancia_id, arquivo_instancia in ids:
        linhas_id = [
            l for l in linhas_gerais
            if l["instancia"] == instancia_id and l["arquivo_instancia"] == arquivo_instancia
        ]
        fos_g = [float(l["grasp_fo"]) for l in linhas_id if l["grasp_fo"]]
        fos_dm = [float(l["dmgrasp_fo"]) for l in linhas_id if l["dmgrasp_fo"]]
        exato_vals = [float(l["exato_fo"]) for l in linhas_id if l["exato_fo"]]
        tempo_ex = [float(l["exato_tempo_s"]) for l in linhas_id if l["exato_tempo_s"]]

        exato_str = f"{exato_vals[0]:.2f}" if exato_vals else "—"
        tempo_ex_str = f"{tempo_ex[0]:.1f}" if tempo_ex else "—"

        if exato_vals:
            ref = exato_vals[0]
            gap_g = f"{100.0 * (ref - np.mean(fos_g)) / ref:.2f}" if fos_g else "—"
            gap_dm = f"{100.0 * (ref - np.mean(fos_dm)) / ref:.2f}" if fos_dm else "—"
        else:
            gap_g = "—"
            gap_dm = "—"

        rotulo = f"{instancia_id}/{arquivo_instancia}"
        print(fmt.format(
            rotulo[:34],
            exato_str,
            tempo_ex_str,
            f"{np.mean(fos_g):.2f} ± {np.std(fos_g):.2f}" if fos_g else "—",
            f"{np.mean(fos_dm):.2f} ± {np.std(fos_dm):.2f}" if fos_dm else "—",
            gap_g,
            gap_dm,
        ))

    print("=" * 90)
    print("\nExperimentos concluídos. Arquivos em:", PASTA_RESULTADOS)


# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sem-exato", action="store_true",
        help="Pula o modelo exato, útil sem licença Gurobi."
    )
    parser.add_argument(
        "--instancias", nargs="+", type=int, default=None,
        help="Roda apenas os tamanhos informados, ex: --instancias 50 100."
    )
    parser.add_argument(
        "--tempo-limite", type=int, default=None,
        help="Tempo limite por execução em segundos, ex: 3600 = 1h. Sobrescreve TEMPO_LIMITE_S."
    )
    parser.add_argument(
        "--clusters", nargs="+", type=int, default=None,
        help="Filtra quantidades de clusters, ex: --clusters 3 5 10."
    )
    parser.add_argument(
        "--desvios-cluster-km", nargs="+", type=float, default=None,
        help="Filtra desvios dos clusters em km, ex: --desvios-cluster-km 1.0 2.0."
    )
    args = parser.parse_args()

    if args.tempo_limite is not None:
        TEMPO_LIMITE_S = args.tempo_limite

    main(
        sem_exato=args.sem_exato,
        tamanhos_filtro=args.instancias,
        clusters_filtro=args.clusters,
        desvios_filtro=args.desvios_cluster_km,
    )
