import os
from openai import OpenAI
from dotenv import load_dotenv
from .base import LLMInterface

# Carica le variabili dal file .env (es. OPENAI_API_KEY)
load_dotenv()

class OpenAILLM(LLMInterface):
    """
    Wrapper per i modelli OpenAI (es. gpt-4o, gpt-3.5-turbo).
    """
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.7):
        super().__init__(model_name)
        # Se non trova la chiave, lo segnaliamo ma non rompiamo tutto subito
        # (utile se stiamo usando solo il Mock)
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = None
            print("⚠️ ATTENZIONE: OPENAI_API_KEY mancante. OpenAILLM non funzionerà.")

        self.default_temp = temperature

    def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """
        Chiama l'API di OpenAI.
        """
        if not self.client:
             return "ERRORE: Chiave API mancante. Configura il file .env"

        # Costruiamo la lista dei messaggi
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            # Chiamata API reale
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=kwargs.get("temperature", self.default_temp),
                max_tokens=kwargs.get("max_tokens", 500)
            )
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error generating response: {str(e)}"