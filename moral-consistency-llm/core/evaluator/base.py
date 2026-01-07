from abc import ABC, abstractmethod
from typing import Dict, Any

class EvaluatorInterface(ABC):
    """
    Interfaccia base per i sistemi di valutazione.
    """
    
    @abstractmethod
    def evaluate(self, text: str) -> Dict[str, Any]:
        """
        Analizza il testo e restituisce metriche (es. is_refusal, confidence).
        """
        pass