import os
from datetime import datetime
from tqdm import tqdm
from core.models.openai_llm import OpenAILLM
from core.models.groq_llm import GroqLLM
from core.prompts.injector import PromptInjector
from data.loader import DataLoader
from data.schemas import ModelResponse

# Mapping Nomi Puliti
MODEL_ALIASES = {
    "gpt-3.5-turbo": "Closed_GPT3.5",
    "llama-3.1-8b-instant": "Open_Llama3_Small",
    "llama-3.3-70b-versatile": "Open_Llama3_Big"
}

class ExperimentRunner:
    def __init__(self, llm_instance):
        self.llm = llm_instance
        self.clean_name = MODEL_ALIASES.get(self.llm.model_name, "Unknown_Model")
        self.output_dir = "results/1_raw"
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self, limit=5):
        # Carichiamo i dati col nuovo Loader sicuro
        scenarios = DataLoader.load_ethics_commonsense(limit=limit)
        
        filename = f"{self.output_dir}/{self.clean_name}.jsonl"
        if os.path.exists(filename):
            os.remove(filename)
        
        print(f"\n🚀 Test Modello: {self.clean_name}...")
        
        for scenario in tqdm(scenarios, desc="Generating"):
            for style in ["stoic", "anxious", "authoritative"]:
                
                # Debug: vediamo se il testo c'è
                if not scenario.text:
                    print("⚠️ ERRORE: Testo scenario vuoto!")
                    continue

                full_prompt = PromptInjector.apply_template(scenario.text, style)
                
                # Generazione Protetta (Try/Except)
                try:
                    response_text = self.llm.generate(full_prompt)
                except Exception as e:
                    response_text = f"SYSTEM_ERROR: {str(e)}"

                result = ModelResponse(
                    scenario_id=scenario.id,
                    model_name=self.clean_name,
                    prompt_style=style,
                    raw_text=response_text
                )
                
                with open(filename, "a") as f:
                    f.write(result.model_dump_json() + "\n")
                    
        print(f"✅ Salvato: {filename}")

if __name__ == "__main__":
    # LISTA MODELLI AGGIORNATA (Senza Gemma)
    models_to_test = [
        # Se non hai pagato OpenAI, questo fallirà (darà SYSTEM_ERROR nel file)
        OpenAILLM("gpt-3.5-turbo"), 
        
        # Modelli Groq (Funzionanti)
        GroqLLM("llama-3.1-8b-instant"),      
        GroqLLM("llama-3.3-70b-versatile")
    ]

    print(f"🏁 Inizio Benchmark (3 Modelli)...")
    for model in models_to_test:
        runner = ExperimentRunner(llm_instance=model)
        runner.run(limit=5)