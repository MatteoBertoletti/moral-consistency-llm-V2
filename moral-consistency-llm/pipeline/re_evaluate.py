import json
import glob
import os
from tqdm import tqdm
from core.models.openai_llm import OpenAILLM
from core.evaluator.llm_judge import LLMJudge

def run_re_evaluation():
    print("⚖️ Attivazione Giudice (GPT-3.5)...")
    judge_model = OpenAILLM("gpt-3.5-turbo", temperature=0.0)
    judge = LLMJudge(judge_model)
    
    # Prende i file dalla cartella 1_raw
    input_files = glob.glob("results/1_raw/*.jsonl")
    output_dir = "results/2_scored"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📂 Trovati {len(input_files)} file grezzi da valutare.")
    
    for filepath in input_files:
        filename = os.path.basename(filepath) # es. "Closed_GPT3.5.jsonl"
        output_path = os.path.join(output_dir, filename) # es. "results/2_scored/Closed_GPT3.5.jsonl"
        
        print(f"\n🔍 Valutazione: {filename}")
        
        new_data = []
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        for line in tqdm(lines, desc="Judging"):
            item = json.loads(line)
            # Contesto per il giudice
            input_ctx = f"Style: {item['prompt_style']} - Scenario: {item['scenario_id']}"
            verdict = judge.evaluate(item['raw_text'], input_prompt=input_ctx)
            
            item['is_refusal'] = verdict['is_refusal']
            item['is_sycophantic'] = verdict['is_sycophantic']
            item['judge_reasoning'] = verdict['reasoning']
            new_data.append(item)
            
        with open(output_path, 'w') as f_out:
            for entry in new_data:
                f_out.write(json.dumps(entry) + "\n")
                
        print(f"✅ Voti salvati in: {output_path}")

if __name__ == "__main__":
    run_re_evaluation()