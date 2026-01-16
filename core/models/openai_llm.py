import os
from openai import OpenAI
from dotenv import load_dotenv
from core.models.base import BaseLLM

load_dotenv()

class OpenAILLM(BaseLLM):
    def __init__(self, model_name="gpt-4o-mini"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = model_name

    def generate_with_probs(self, system_prompt, user_prompt):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            logprobs=True, # Fondamentale per la Softmax
            top_logprobs=5, # Vediamo i 5 token più probabili
            max_tokens=150,
            temperature=0 # Massima determinismo per coerenza
        )
        
        content = response.choices[0].message.content
        # Estraiamo i logprobs del primo token (solitamente la risposta Yes/No)
        probs_data = response.choices[0].logprobs.content[0].top_logprobs
        
        return content, probs_data