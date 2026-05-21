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

