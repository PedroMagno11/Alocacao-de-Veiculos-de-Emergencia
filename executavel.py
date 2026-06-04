from grasp.grasp import executar_grasp
from dm_grasp.dm_grasp import executar_dm_grasp
from instances.read_instance import ler_instancia
from config.PARAMETROS import (
    MAX_ITERACOES, MAX_ITERACOES_SEM_MELHORA,
    PARAMETRO_ALPHA, QUANTIDADE_MAXIMA_POR_TIPO,
    SEMENTE_ALEATORIA, TIPOS_AMBULANCIA,
)

# dataframe = ler_instancia("instances/instancia.csv")
dataframe = ler_instancia("instances/instancia_aleatoria_01_10000p.csv")

res_grasp = executar_grasp(
    dataframe                  = dataframe,
    tipos_ambulancia           = TIPOS_AMBULANCIA,
    quantidade_maxima_por_tipo = QUANTIDADE_MAXIMA_POR_TIPO,
    parametro_alpha            = PARAMETRO_ALPHA,
    max_iteracoes              = MAX_ITERACOES,
    max_iteracoes_sem_melhora  = MAX_ITERACOES_SEM_MELHORA,
    semente_aleatoria          = SEMENTE_ALEATORIA,
)

res_dm = executar_dm_grasp(
    dataframe                  = dataframe,
    tipos_ambulancia           = TIPOS_AMBULANCIA,
    quantidade_maxima_por_tipo = QUANTIDADE_MAXIMA_POR_TIPO,
    parametro_alpha            = PARAMETRO_ALPHA,
    max_iteracoes              = MAX_ITERACOES,
    max_iteracoes_sem_melhora  = MAX_ITERACOES_SEM_MELHORA,
    semente_aleatoria          = SEMENTE_ALEATORIA,
)

