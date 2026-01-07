import pandas as pd
import glob
import json
import os

def generate_final_report():
    # Legge dalla cartella 2_scored
    scored_files = glob.glob("results/2_scored/*.jsonl")
    output_dir = "results/3_final_report"
    os.makedirs(output_dir, exist_ok=True)

    if not scored_files:
        print("❌ Nessun file votato trovato in results/2_scored/")
        return

    all_data = []
    for filepath in scored_files:
        with open(filepath, 'r') as f:
            for line in f:
                data = json.loads(line)
                all_data.append({
                    "Model": data['model_name'], # Usa il nome pulito (es. Open_Llama3)
                    "Style": data['prompt_style'],
                    "Refusal": 1 if data['is_refusal'] else 0
                })

    df = pd.DataFrame(all_data)

    # Tabella 1: Refusal Rate
    pivot = df.pivot_table(index="Model", columns="Style", values="Refusal", aggfunc="mean") * 100
    
    print("\n" + "="*50)
    print("🏆 REPORT FINALE (Refusal Rate %)")
    print("="*50)
    print(pivot.round(1))
    
    # Salva CSV
    csv_path = f"{output_dir}/FINAL_SUMMARY.csv"
    pivot.to_csv(csv_path)
    print(f"\n✅ File Excel salvato in: {csv_path}")

if __name__ == "__main__":
    generate_final_report()