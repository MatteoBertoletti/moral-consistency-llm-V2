# core/prompts/injector.py
import pandas as pd
import os
from .templates import WRAPPERS

def run_injection():
    # Carichiamo i tuoi 1000 scenari bilanciati
    input_path = 'results/1_raw/ethics_1000_balanced.csv'
    if not os.path.exists(input_path):
        print(f"❌ Error: {input_path} not found.")
        return

    df = pd.read_csv(input_path)
    wrapped_data = []

    print(f"🚀 Injecting emotional wrappers into {len(df)} scenarios...")

    for _, row in df.iterrows():
        for w_type, template in WRAPPERS.items():
            wrapped_data.append({
                'category': row['category'],
                'ground_truth': row['ground_truth'],
                'wrapper_type': w_type,
                'prompt': template.format(scenario=row['scenario']),
                'original_scenario': row['scenario']
            })

    output_df = pd.DataFrame(wrapped_data)
    output_path = 'results/1_raw/ethics_3000_wrapped.csv'
    output_df.to_csv(output_path, index=False)
    
    print(f"✅ Injection complete! Created {len(output_df)} total prompts.")

if __name__ == "__main__":
    run_injection()