import os
from openai import OpenAI
from dotenv import load_dotenv
from core.models.base import BaseLLM

load_dotenv()

class LlamaLLM(BaseLLM):
    def __init__(self, model_name="meta-llama/llama-3.1-8b-instruct"):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
        self.model_name = model_name

    def generate_with_probs(self, system_prompt, user_prompt):
        """Implementazione obbligatoria del metodo astratto."""
        
        # Forziamo Cerebras e impostiamo temperatura > 0 per i logprobs
        extra_body = {
            "provider": {
                "order": ["Cerebras"],
                "allow_fallbacks": False
            }
        }

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            extra_headers={
                "HTTP-Referer": "http://localhost:3000",
                "X-Title": "Moral Consistency Research",
            },
            extra_body=extra_body,
            logprobs=True,
            top_logprobs=5,
            temperature=1.0, # Cruciale per Cerebras
            max_tokens=150
        )
        
        content = response.choices[0].message.content
        log_obj = response.choices[0].logprobs

        probs_data = []
        if log_obj and hasattr(log_obj, 'content') and log_obj.content:
            # Estraiamo i primi 5 candidati del primo token generato
            probs_data = log_obj.content[0].top_logprobs
        
        return content, probs_data