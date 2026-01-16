import pandas as pd
import re
import os

# 1. Ricerca del file
file_path = 'results/pilot_150_scenarios.csv'
if not os.path.exists(file_path):
    file_path = '../results/pilot_150_scenarios.csv'

if not os.path.exists(file_path):
    print(f"❌ File non trovato in: {file_path}")
    exit()

df = pd.read_csv(file_path)

# 2. Funzione per estrarre le percentuali dalle stringhe Decision_Stats
# Esempio stringa: "Moral (88.1%) | Immoral (11.9%)"
def extract_percentages(stat_str):
    try:
        match = re.search(r'Moral \((\d+\.?\d*)%\) \| Immoral \((\d+\.?\d*)%\)', str(stat_str))
        if match:
            return float(match.group(1)), float(match.group(2))
    except:
        pass
    return 0.0, 0.0

# 3. Estrazione dati per OpenAI e Llama
df[['OA_Moral_%', 'OA_Immoral_%']] = df['OA_Decision_Stats'].apply(lambda x: pd.Series(extract_percentages(x)))
df[['LL_Moral_%', 'LL_Immoral_%']] = df['LL_Decision_Stats'].apply(lambda x: pd.Series(extract_percentages(x)))

# 4. Creazione della Tabella Unica
# Raggruppiamo per Categoria e Wrapper e calcoliamo la media delle percentuali
summary_table = df.groupby(['Category', 'Wrapper'])[[
    'OA_Moral_%', 'OA_Immoral_%', 
    'LL_Moral_%', 'LL_Immoral_%'
]].mean()

# 5. Stampa a terminale
print("\n" + "="*85)
print("📊 ANALISI DISTRIBUZIONE MEDIA LOGPROBS (MORAL vs IMMORAL)")
print("="*85)
print(summary_table.round(2).to_string())
print("="*85)
print("Nota: I valori rappresentano la probabilità media (%) assegnata dai modelli.")

# 6. Salvataggio
summary_table.to_csv('results/logprob_distribution_summary.csv')