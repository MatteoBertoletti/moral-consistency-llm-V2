from abc import ABC, abstractmethod

class BaseLLM(ABC):
    @abstractmethod
    def generate_with_probs(self, system_prompt, user_prompt):
        """Metodo obbligatorio per estrarre testo e probabilità dei token."""
        pass