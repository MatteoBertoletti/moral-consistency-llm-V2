from pydantic import BaseModel, Field
from typing import Optional, List, Literal

class MoralScenario(BaseModel):
    """
    Rappresenta un singolo scenario etico normalizzato.
    Corrisponde alla specifica del Design Document.
    """
    id: str = Field(..., description="Identificativo unico dello scenario")
    text: str = Field(..., description="Il testo del dilemma o della domanda")
    source_dataset: Literal["ethics", "social_chemistry", "custom"]
    label: str = Field(..., description="Ground truth: unacceptable, acceptable, etc.")
    
    # Metadati opzionali per analisi future
    difficulty: Optional[str] = "unknown"

class ModelResponse(BaseModel):
    """
    Cattura l'output del modello per un dato scenario.
    """
    scenario_id: str
    model_name: str
    prompt_style: Literal["stoic", "anxious", "authoritative"]
    raw_text: str
    
    # Metriche calcolate successivamente
    is_refusal: bool = False
    refusal_confidence: float = 0.0