"""
Executável DM-GRASP para o PMCS-FA.

Este script permite rodar o DM-GRASP separadamente em uma ou várias instâncias,
salvando automaticamente resultados e figuras.

Estrutura esperada das instâncias:

    instances/100p/8_clusters/desvio_1_0km/*.csv

Saída gerada:

    results/dm_grasp/100p/8_clusters/desvio_1_0km/<nome_instancia>/
        resultados_dm_grasp.csv
        solucao_dm_grasp.csv
        fig1_convergencia.png
        fig2_antes_apos_bl.png

    results/dm_grasp/consolidado/
        resultados_dm_grasp_geral.csv

Exemplos:

    python -m dm_grasp.dm_grasp_experimentos

    python -m dm_grasp.dm_grasp_experimentos --instancias 100 500

    python -m dm_grasp.dm_grasp_experimentos --instancias 100 --clusters 8 --desvios-cluster-km 1.0

    python -m dm_grasp.dm_grasp_experimentos --arquivo instances/100p/8_clusters/desvio_1_0km/instancia_aleatoria_01_100p_8c_desvio_1_0km.csv
"""

import argparse
import csv
import re
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config.PARAMETROS import (
    PARAMETRO_ALPHA,
    QUANTIDADE_MAXIMA_POR_TIPO,
    MAX_ITERACOES_SEM_MELHORA,
    TIPOS_AMBULANCIA,
    SEMENTE_ALEATORIA,
)

from instances.read_instance import ler_instancia
from dm_grasp.dm_grasp import executar_dm_grasp


# ──────────────────────────────────────────────────────────────
# Parâmetros padrão
# ──────────────────────────────────────────────────────────────

PASTA_INSTANCIAS = Path("instances")
PASTA_RESULTADOS = Path("results") / "dm_grasp"

SEMENTES = [SEMENTE_ALEATORIA]
TAMANHOS = [100, 500, 1000, 1500, 2000]

ITERACOES_FASE1 = 150
ITERACOES_FASE2 = 150
TAMANHO_MEMORIA_ELITE = 40
FREQUENCIA_MINIMA = 0.3
TEMPO_LIMITE_S = 3600
PROPORCAO_TEMPO_FASE1 = 0.4


# ──────────────────────────────────────────────────────────────
# Estrutura de instância
# ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class InstanciaExperimento:
    caminho: Path
    tamanho: int
    clusters: int | None
    desvio_cluster_km: float | None

    @property
    def nome_instancia(self) -> str:
        return self.caminho.stem

    @property
    def id_experimento(self) -> str:
        partes = [f"{self.tamanho}p"]

        if self.clusters is not None:
            partes.append(f"{self.clusters}c")

        if self.desvio_cluster_km is not None:
            partes.append(f"desvio_{formatar_desvio(self.desvio_cluster_km)}km")

        return "_".join(partes)


def formatar_desvio(valor: float) -> str:
    return str(valor).replace(".", "_")


def extrair_primeiro_inteiro_padrao(texto: str, padrao: str) -> int | None:
    match = re.search(padrao, texto, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def extrair_desvio(texto: str) -> float | None:
    match = re.search(
        r"desvio[_-]?(\d+(?:[._]\d+)?)\s*km?",
        texto,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    return float(match.group(1).replace("_", "."))


def descobrir_instancias(
    pasta_instancias: Path,
    tamanhos_filtro: list[int] | None = None,
    clusters_filtro: list[int] | None = None,
    desvios_filtro: list[float] | None = None,
) -> list[InstanciaExperimento]:
    tamanhos_set = set(tamanhos_filtro) if tamanhos_filtro else None
    clusters_set = set(clusters_filtro) if clusters_filtro else None
    desvios_set = {round(v, 6) for v in desvios_filtro} if desvios_filtro else None

    instancias = []

    for caminho in pasta_instancias.rglob("*.csv"):
        relativo = caminho.relative_to(pasta_instancias)
        partes = relativo.parts
        texto_relativo = str(relativo)

        tamanho = None
        clusters = None
        desvio = None

        for parte in partes:
            tamanho = tamanho or extrair_primeiro_inteiro_padrao(parte, r"(\d+)p")
            clusters = clusters or extrair_primeiro_inteiro_padrao(parte, r"(\d+)[_-]?clusters?")
            desvio = desvio if desvio is not None else extrair_desvio(parte)

        tamanho = tamanho or extrair_primeiro_inteiro_padrao(caminho.name, r"_(\d+)p\.csv$")
        tamanho = tamanho or extrair_primeiro_inteiro_padrao(texto_relativo, r"(\d+)p")
        clusters = clusters or extrair_primeiro_inteiro_padrao(texto_relativo, r"(\d+)c(?:_|\b)")
        desvio = desvio if desvio is not None else extrair_desvio(texto_relativo)

        if tamanho is None:
            print(f"[AVISO] Ignorando CSV sem tamanho identificável: {caminho}")
            continue

        if tamanhos_set and tamanho not in tamanhos_set:
            continue

        if clusters_set and clusters not in clusters_set:
            continue

        if desvios_set and (desvio is None or round(desvio, 6) not in desvios_set):
            continue

        instancias.append(
            InstanciaExperimento(
                caminho=caminho,
                tamanho=tamanho,
                clusters=clusters,
                desvio_cluster_km=desvio,
            )
        )

    instancias.sort(
        key=lambda i: (
            i.tamanho,
            i.clusters if i.clusters is not None else -1,
            i.desvio_cluster_km if i.desvio_cluster_km is not None else -1,
            str(i.caminho),
        )
    )

    return instancias


def criar_instancia_unica(caminho: Path) -> InstanciaExperimento:
    texto = str(caminho)

    tamanho = extrair_primeiro_inteiro_padrao(texto, r"(\d+)p")
    clusters = extrair_primeiro_inteiro_padrao(texto, r"(\d+)[_-]?clusters?")
    desvio = extrair_desvio(texto)

    if tamanho is None:
        dataframe = ler_instancia(str(caminho))
        tamanho = len(dataframe)

    return InstanciaExperimento(
        caminho=caminho,
        tamanho=tamanho,
        clusters=clusters,
        desvio_cluster_km=desvio,
    )


# ──────────────────────────────────────────────────────────────
# Pastas e CSV
# ──────────────────────────────────────────────────────────────

def obter_pasta_cenario(instancia: InstanciaExperimento) -> Path:
    try:
        relativo_pai = instancia.caminho.parent.relative_to(PASTA_INSTANCIAS)
    except ValueError:
        relativo_pai = Path("manual") / instancia.id_experimento

    if str(relativo_pai) == ".":
        relativo_pai = Path("formato_antigo") / instancia.id_experimento

    pasta = PASTA_RESULTADOS / relativo_pai
    pasta.mkdir(parents=True, exist_ok=True)
    return pasta


def obter_pasta_instancia(instancia: InstanciaExperimento) -> Path:
    pasta = obter_pasta_cenario(instancia) / instancia.nome_instancia
    pasta.mkdir(parents=True, exist_ok=True)
    return pasta


def escrever_csv(caminho: Path, cabecalho: list[str], linhas: list[dict]) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)

    with open(caminho, "w", newline="", encoding="utf-8") as arquivo:
        writer = csv.DictWriter(arquivo, fieldnames=cabecalho)
        writer.writeheader()
        writer.writerows(linhas)

    print(f"  Salvo: {caminho}")


def salvar_solucao(caminho: Path, solucao, tipos_ambulancia) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)

    with open(caminho, "w", newline="", encoding="utf-8") as arquivo:
        writer = csv.DictWriter(
            arquivo,
            fieldnames=["regiao", "tipo", "nome_tipo"],
        )
        writer.writeheader()

        for regiao, tipo in sorted(solucao):
            writer.writerow({
                "regiao": regiao,
                "tipo": tipo,
                "nome_tipo": tipos_ambulancia[tipo]["nome"],
            })

    print(f"  Salvo: {caminho}")


# ──────────────────────────────────────────────────────────────
# Figuras
# ──────────────────────────────────────────────────────────────

def salvar_fig1_convergencia(hist_f1, hist_f2, titulo, pasta):
    fig, ax = plt.subplots(figsize=(11, 5))

    n1 = len(hist_f1)
    todos = hist_f1 + hist_f2

    iter_f1 = list(range(1, len(hist_f1) + 1))
    iter_f2 = list(range(len(hist_f1) + 1, len(hist_f1) + len(hist_f2) + 1))

    ax.scatter(iter_f1, hist_f1, s=22, alpha=0.65, label="GRASP puro (Fase 1)")
    ax.scatter(iter_f2, hist_f2, s=22, alpha=0.65, label="DM-GRASP (Fase 2)")

    if todos:
        melhor_acumulado = np.maximum.accumulate(todos)
        ax.plot(
            range(1, len(todos) + 1),
            melhor_acumulado,
            linewidth=1.8,
            linestyle="--",
            label="Melhor acumulado",
        )

    if n1 > 0:
        ax.axvline(x=n1 + 0.5, linestyle=":", linewidth=1.2)

    ax.set_title(f"Evolução da Função Objetivo — {titulo}")
    ax.set_xlabel("Iteração")
    ax.set_ylabel("Função Objetivo")
    ax.legend()
    ax.grid(True, alpha=0.25, linestyle="--")

    fig.tight_layout()

    caminho = pasta / "fig1_convergencia.png"
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Salvo: {caminho}")


def salvar_fig2_antes_apos_bl(
    hist_antes_dm,
    hist_apos_dm,
    titulo,
    pasta,
):
    fig, ax = plt.subplots(figsize=(11, 5))

    iteracoes = range(1, len(hist_antes_dm) + 1)

    ax.scatter(iteracoes, hist_antes_dm, s=22, alpha=0.45, label="Pré-BL")
    ax.scatter(iteracoes, hist_apos_dm, s=22, alpha=0.85, label="Pós-BL")

    if hist_apos_dm:
        melhor_acumulado = np.maximum.accumulate(hist_apos_dm)
        ax.plot(
            iteracoes,
            melhor_acumulado,
            linewidth=1.8,
            linestyle="--",
            label="Melhor acumulado",
        )

    ax.set_title(f"FO antes e após Busca Local — {titulo}")
    ax.set_xlabel("Iteração")
    ax.set_ylabel("Função Objetivo")
    ax.legend()
    ax.grid(True, alpha=0.25, linestyle="--")

    fig.tight_layout()

    caminho = pasta / "fig2_antes_apos_bl.png"
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Salvo: {caminho}")


def salvar_fig3_boxplot(dados, pasta, nome_arquivo="fig3_boxplot.png", titulo="Distribuição das soluções — DM-GRASP"):
    if not dados:
        return

    labels = list(dados.keys())
    valores = [dados[label] for label in labels]

    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(labels)), 5))

    ax.boxplot(valores, tick_labels=labels, patch_artist=True)
    ax.set_title(titulo)
    ax.set_ylabel("Função Objetivo")
    ax.grid(True, alpha=0.3)

    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()

    caminho = pasta / nome_arquivo
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Salvo: {caminho}")


# ──────────────────────────────────────────────────────────────
# Execução
# ──────────────────────────────────────────────────────────────

def executar_para_instancia(
    instancia: InstanciaExperimento,
    sementes: list[int],
    parametro_alpha: float,
    iteracoes_fase1: int,
    iteracoes_fase2: int,
    max_iteracoes_sem_melhora,
    tamanho_memoria_elite: int,
    frequencia_minima: float,
    tempo_limite_s: float | None,
    proporcao_tempo_fase1: float,
) -> tuple[list[dict], list[float]]:
    pasta_saida = obter_pasta_instancia(instancia)

    dataframe = ler_instancia(str(instancia.caminho))
    total_pontos = len(dataframe)

    print("\n" + "=" * 70)
    print(f"  DM-GRASP — {instancia.id_experimento}")
    print(f"  Instância : {instancia.nome_instancia}")
    print(f"  Arquivo   : {instancia.caminho}")
    print(f"  Pontos    : {total_pontos}")
    print(f"  Clusters  : {instancia.clusters if instancia.clusters is not None else '—'}")
    print(f"  Desvio    : {instancia.desvio_cluster_km if instancia.desvio_cluster_km is not None else '—'} km")
    print(f"  Saída     : {pasta_saida}")
    print("=" * 70)

    cabecalho = [
        "metodo",
        "instancia",
        "arquivo_instancia",
        "caminho_instancia",
        "n_pontos",
        "clusters",
        "desvio_cluster_km",
        "semente",
        "fo",
        "tempo_s",
        "iteracoes_fase1",
        "iteracoes_fase2",
        "melhor_fo_fase1",
        "melhor_fo_fase2",
    ]

    linhas = []
    fos = []

    historicos_fig = None
    melhor_solucao_global = None
    melhor_fo_global = -1e18

    for semente in sementes:
        print(f"\n  Semente {semente}...")

        inicio = time.perf_counter()

        (
            melhor_solucao,
            melhor_fo,
            _pontos_cobertos,
            historico_fo_f1,
            historico_fo_f2,
            historico_antes_bl_f1,
            historico_apos_bl_f1,
            historico_antes_bl_f2,
            historico_apos_bl_f2,
        ) = executar_dm_grasp(
            dataframe=dataframe,
            tipos_ambulancia=TIPOS_AMBULANCIA,
            quantidade_maxima_por_tipo=QUANTIDADE_MAXIMA_POR_TIPO,
            parametro_alpha=parametro_alpha,
            iteracoes_fase1=iteracoes_fase1,
            iteracoes_fase2=iteracoes_fase2,
            max_iteracoes_sem_melhora=max_iteracoes_sem_melhora,
            semente_aleatoria=semente,
            tamanho_memoria_elite=tamanho_memoria_elite,
            frequencia_minima=frequencia_minima,
            tempo_limite_s=tempo_limite_s,
            proporcao_tempo_fase1=proporcao_tempo_fase1,
        )

        tempo = time.perf_counter() - inicio
        fos.append(melhor_fo)

        melhor_fo_f1 = max(historico_fo_f1) if historico_fo_f1 else ""
        melhor_fo_f2 = max(historico_fo_f2) if historico_fo_f2 else ""

        linha = {
            "metodo": "DM-GRASP",
            "instancia": instancia.id_experimento,
            "arquivo_instancia": instancia.nome_instancia,
            "caminho_instancia": str(instancia.caminho),
            "n_pontos": total_pontos,
            "clusters": instancia.clusters if instancia.clusters is not None else "",
            "desvio_cluster_km": instancia.desvio_cluster_km if instancia.desvio_cluster_km is not None else "",
            "semente": semente,
            "fo": f"{melhor_fo:.4f}",
            "tempo_s": f"{tempo:.4f}",
            "iteracoes_fase1": len(historico_fo_f1),
            "iteracoes_fase2": len(historico_fo_f2),
            "melhor_fo_fase1": f"{melhor_fo_f1:.4f}" if melhor_fo_f1 != "" else "",
            "melhor_fo_fase2": f"{melhor_fo_f2:.4f}" if melhor_fo_f2 != "" else "",
        }

        linhas.append(linha)

        if melhor_fo > melhor_fo_global:
            melhor_fo_global = melhor_fo
            melhor_solucao_global = melhor_solucao

        if historicos_fig is None:
            historicos_fig = {
                "hist_fo_f1": historico_fo_f1,
                "hist_fo_f2": historico_fo_f2,
                "hist_antes": historico_antes_bl_f1 + historico_antes_bl_f2,
                "hist_apos": historico_apos_bl_f1 + historico_apos_bl_f2,
                "semente": semente,
            }

        print(f"  Resultado seed {semente}: FO={melhor_fo:.2f} | tempo={tempo:.2f}s")

    escrever_csv(pasta_saida / "resultados_dm_grasp.csv", cabecalho, linhas)

    if melhor_solucao_global is not None:
        salvar_solucao(
            pasta_saida / "solucao_dm_grasp.csv",
            melhor_solucao_global,
            TIPOS_AMBULANCIA,
        )

    if historicos_fig:
        titulo = f"{instancia.id_experimento} / {instancia.nome_instancia} / seed {historicos_fig['semente']}"
        salvar_fig1_convergencia(
            historicos_fig["hist_fo_f1"],
            historicos_fig["hist_fo_f2"],
            titulo,
            pasta_saida,
        )
        salvar_fig2_antes_apos_bl(
            historicos_fig["hist_antes"],
            historicos_fig["hist_apos"],
            titulo,
            pasta_saida,
        )

    salvar_fig3_boxplot(
        {instancia.nome_instancia: fos},
        pasta_saida,
        nome_arquivo="fig3_boxplot.png",
        titulo=f"Distribuição das soluções — {instancia.id_experimento} / {instancia.nome_instancia}",
    )

    print(f"\n  Resumo {instancia.nome_instancia}:")
    print(f"    média={np.mean(fos):.2f}")
    print(f"    dp={np.std(fos):.2f}")
    print(f"    melhor={max(fos):.2f}")
    print(f"    pior={min(fos):.2f}")

    return linhas, fos


def main():
    parser = argparse.ArgumentParser(
        description="Executa DM-GRASP para uma ou várias instâncias do PMCS-FA."
    )

    parser.add_argument(
        "--arquivo",
        type=str,
        default=None,
        help="Executa uma única instância específica.",
    )

    parser.add_argument(
        "--instancias",
        nargs="+",
        type=int,
        default=None,
        help="Filtra tamanhos. Ex: --instancias 100 500.",
    )

    parser.add_argument(
        "--clusters",
        nargs="+",
        type=int,
        default=None,
        help="Filtra quantidades de clusters. Ex: --clusters 8 12.",
    )

    parser.add_argument(
        "--desvios-cluster-km",
        nargs="+",
        type=float,
        default=None,
        help="Filtra desvios. Ex: --desvios-cluster-km 1.0 1.5.",
    )

    parser.add_argument(
        "--sementes",
        nargs="+",
        type=int,
        default=None,
        help="Seeds. Ex: --sementes 42 43 44.",
    )

    parser.add_argument(
        "--tempo-limite",
        type=float,
        default=TEMPO_LIMITE_S,
        help="Tempo limite por execução, em segundos.",
    )

    parser.add_argument(
        "--sem-tempo-limite",
        action="store_true",
        help="Executa apenas por número de iterações, sem limite de tempo.",
    )

    parser.add_argument(
        "--iteracoes-fase1",
        type=int,
        default=ITERACOES_FASE1,
    )

    parser.add_argument(
        "--iteracoes-fase2",
        type=int,
        default=ITERACOES_FASE2,
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=PARAMETRO_ALPHA,
    )

    parser.add_argument(
        "--tamanho-elite",
        type=int,
        default=TAMANHO_MEMORIA_ELITE,
    )

    parser.add_argument(
        "--frequencia-minima",
        type=float,
        default=FREQUENCIA_MINIMA,
    )

    args = parser.parse_args()

    sementes = args.sementes if args.sementes else SEMENTES
    tempo_limite = None if args.sem_tempo_limite else args.tempo_limite

    if args.arquivo:
        instancias = [criar_instancia_unica(Path(args.arquivo))]
    else:
        tamanhos = args.instancias if args.instancias is not None else TAMANHOS

        instancias = descobrir_instancias(
            PASTA_INSTANCIAS,
            tamanhos_filtro=tamanhos,
            clusters_filtro=args.clusters,
            desvios_filtro=args.desvios_cluster_km,
        )

    if not instancias:
        print("[AVISO] Nenhuma instância encontrada.")
        return

    cabecalho = [
        "metodo",
        "instancia",
        "arquivo_instancia",
        "caminho_instancia",
        "n_pontos",
        "clusters",
        "desvio_cluster_km",
        "semente",
        "fo",
        "tempo_s",
        "iteracoes_fase1",
        "iteracoes_fase2",
        "melhor_fo_fase1",
        "melhor_fo_fase2",
    ]

    linhas_gerais = []
    boxplot_geral = {}

    for instancia in instancias:
        linhas, fos = executar_para_instancia(
            instancia=instancia,
            sementes=sementes,
            parametro_alpha=args.alpha,
            iteracoes_fase1=args.iteracoes_fase1,
            iteracoes_fase2=args.iteracoes_fase2,
            max_iteracoes_sem_melhora=MAX_ITERACOES_SEM_MELHORA,
            tamanho_memoria_elite=args.tamanho_elite,
            frequencia_minima=args.frequencia_minima,
            tempo_limite_s=tempo_limite,
            proporcao_tempo_fase1=PROPORCAO_TEMPO_FASE1,
        )

        linhas_gerais.extend(linhas)

        chave = f"{instancia.id_experimento}_{instancia.nome_instancia}"
        boxplot_geral[chave] = fos

    pasta_consolidado = PASTA_RESULTADOS / "consolidado"
    pasta_consolidado.mkdir(parents=True, exist_ok=True)

    escrever_csv(
        pasta_consolidado / "resultados_dm_grasp_geral.csv",
        cabecalho,
        linhas_gerais,
    )

    salvar_fig3_boxplot(
        boxplot_geral,
        pasta_consolidado,
        nome_arquivo="fig3_boxplot_dm_grasp_geral.png",
        titulo="Distribuição das soluções — DM-GRASP — Todas as instâncias",
    )

    print("\n" + "=" * 80)
    print("RESUMO GERAL — DM-GRASP")
    print("=" * 80)

    ids = sorted({(l["instancia"], l["arquivo_instancia"]) for l in linhas_gerais})

    for instancia_id, arquivo_instancia in ids:
        linhas_id = [
            l for l in linhas_gerais
            if l["instancia"] == instancia_id
            and l["arquivo_instancia"] == arquivo_instancia
        ]

        fos = [float(l["fo"]) for l in linhas_id]

        print(
            f"{instancia_id}/{arquivo_instancia}: "
            f"média={np.mean(fos):.2f} ± {np.std(fos):.2f} | "
            f"melhor={max(fos):.2f} | pior={min(fos):.2f}"
        )

    print("=" * 80)
    print(f"Arquivos salvos em: {PASTA_RESULTADOS}")


if __name__ == "__main__":
    main()
