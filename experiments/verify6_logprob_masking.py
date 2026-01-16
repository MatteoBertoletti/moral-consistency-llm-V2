import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def run_explainability_analysis(input_path):
    # 1. Caricamento dati
    df = pd.read_csv(input_path)
    
    # Pulizia nomi colonne (rimozione spazi)
    df.columns = [c.strip() for c in df.columns]
    
    # Estrazione del valore numerico di Shift per le Heatmap
    # Trasformiamo stringhe tipo "stoic (80% -> 85%)" in valore assoluto di shift
    def extract_shift(cell, base_val_str):
        try:
            base = float(base_val_str.replace('%', ''))
            new_val = float(cell.split('->')[1].split('%')[0].strip())
            return abs(base - new_val)
        except:
            return 0.0

    for m in ['OpenAI', 'Llama']:
        df[f'{m}_Shift_Val'] = df.apply(lambda r: extract_shift(r[f'{m}_Top1'], r[f'{m}_Base']), axis=1)

    # --- 1. GENERAZIONE HEATMAPS (Mappa della Fragilità Etica) ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    for i, m in enumerate(['OpenAI', 'Llama']):
        pivot = df.pivot_table(index='Wrapper', columns='Category', values=f'{m}_Shift_Val', aggfunc='mean')
        sns.heatmap(pivot, annot=True, cmap='YlOrRd', ax=axes[i], fmt=".1f")
        axes[i].set_title(f'Heatmap Fragilità: {m}\n(Shift medio in punti %)')
    
    plt.tight_layout()
    plt.savefig('results/heatmap_fragility_comparison.png')

    # --- 2. BIAS DISCOVERY (Parole Trigger) ---
    # Contiamo quante volte ogni parola appare come Top 1 Influencer
    def get_top_word(cell):
        return cell.split('(')[0].strip().lower()

    all_bias_data = []
    for m in ['OpenAI', 'Llama']:
        top_words = df[f'{m}_Top1'].apply(get_top_word).value_counts().head(10)
        for word, count in top_words.items():
            all_bias_data.append({'Model': m, 'Word': word, 'Frequency': count})
    
    bias_df = pd.DataFrame(all_bias_data)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=bias_df, x='Word', y='Frequency', hue='Model')
    plt.title('Bias Discovery: Top 10 Token che Manipolano il Modello')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/bias_discovery_tokens.png')

    print("📊 Analisi completata. Generati: heatmap_fragility_comparison.png e bias_discovery_tokens.png")

if __name__ == "__main__":
    run_explainability_analysis('results/word_attribution_balanced_final.csv')