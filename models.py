from dataclasses import dataclass

@dataclass
class Localizacao:
    latitude: float
    longitude: float
    peso: float
    
@dataclass
class Alocacao:
    local_id: str
    latitude: float
    longitude: float
    peso: float
    complexidade: str
    demanda_simples: float
    demanda_complexa: float
    prioridade_aloc_ambulancia_A: float
    prioridade_aloc_ambulancia_B: float



