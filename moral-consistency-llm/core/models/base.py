from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class LLMInterface(ABC):
    """
    Classe Astratta (Abstract Base Class) per i modelli linguistici.
    Definisce il contratto che tutti i modelli devono rispettare.
    """
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.config = kwargs

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """
        Genera una risposta testuale dato un prompt.
        
        Args:
            prompt: Il messaggio dell'utente.
            system_prompt: (Opzionale) Istruzione di sistema.
            **kwargs: Parametri extra.
            
        Returns:
            La stringa generata.
        """
        pass