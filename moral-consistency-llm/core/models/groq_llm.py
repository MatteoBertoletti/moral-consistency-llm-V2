import os
from groq import Groq
from dotenv import load_dotenv
from .base import LLMInterface

load_dotenv()

class GroqLLM(LLMInterface):
    """
    Wrapper per i modelli Open Source ospitati su Groq.
    Supporta: Llama 3, Mixtral, Gemma, ecc.
    """
    
    def __init__(self, model_name: str, temperature: float = 0.7):
        super().__init__(model_name)
        api_key = os.getenv("GROQ_API_KEY")
        
        if api_key:
            self.client = Groq(api_key=api_key)
        else:
            self.client = None
            print("⚠️ ATTENZIONE: GROQ_API_KEY mancante. I modelli Open non funzioneranno.")

        self.default_temp = temperature

    def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        if not self.client:
            return "ERRORE: Chiave Groq mancante."

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            # Chiamata API a Groq
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=kwargs.get("temperature", self.default_temp),
                max_tokens=kwargs.get("max_tokens", 500)
            )
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error Groq: {str(e)}"