import pandas as pd
import os

def load_balanced_ethics_data():
    base_url = "https://huggingface.co/datasets/hendrycks/ethics/resolve/main/data"
    categories = ['commonsense', 'deontology', 'justice', 'utilitarianism', 'virtue']
    all_data = []

    print(f"{'Categoria':<20} | {'Morali':<8} | {'Immorali':<8} | {'Status'}")
    print("-" * 60)

    for cat in categories:
        url = f"{base_url}/{cat}/test.csv"
        try:
            if cat == 'utilitarianism':
                # Utilitarianism NON ha header. Col 0 = Morale, Col 1 = Immorale
                df = pd.read_csv(url, header=None)
                s_moral = df[0].sample(n=100, random_state=42)
                s_immoral = df[1].sample(n=100, random_state=42)
                
                for text in s_moral:
                    all_data.append({'category': cat, 'scenario': str(text).strip(), 'ground_truth': 'Morale'})
                for text in s_immoral:
                    all_data.append({'category': cat, 'scenario': str(text).strip(), 'ground_truth': 'Immorale'})
            
            else:
                # Altre categorie hanno header e colonna 'label'
                df = pd.read_csv(url)
                
                # Filtriamo i due gruppi
                df_moral = df[df['label'] == 0]
                df_immoral = df[df['label'] == 1]
                
                # Campionamento di 100 record per gruppo
                s_moral = df_moral.sample(n=100, random_state=42)
                s_immoral = df_immoral.sample(n=100, random_state=42)
                
                for sub_df, label_str in [(s_moral, "Morale"), (s_immoral, "Immorale")]:
                    for _, row in sub_df.iterrows():
                        # Commonsense usa 'input', gli altri 'scenario'
                        text = row.get('input') or row.get('scenario') or row.get('text')
                        all_data.append({
                            'category': cat,
                            'scenario': str(text).strip(),
                            'ground_truth': label_str
                        })
            
            print(f"{cat:<20} | {100:<8} | {100:<8} | ✅ OK")
            
        except Exception as e:
            print(f"{cat:<20} | ❌ ERRORE: {str(e)[:25]}")

    if all_data:
        os.makedirs('results/1_raw', exist_ok=True)
        output_path = 'results/1_raw/ethics_1000_balanced.csv'
        pd.DataFrame(all_data).to_csv(output_path, index=False)
        print(f"\n🚀 Dataset 1000 record (50/50) creato con successo!")

if __name__ == "__main__":
    load_balanced_ethics_data()