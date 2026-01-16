import pandas as pd
import os

def run_verify_9_clean():
    # Caricamento dati
    file_path = 'results/esperimento_allineamento_30_scenari.csv'
    if not os.path.exists(file_path):
        print(f"❌ Errore: File '{file_path}' non trovato. Esegui prima il Verify 8.")
        return

    df = pd.read_csv(file_path)

    # 1. Pulizia dati (conversione in numeri)
    if df['Approvazione Finale'].dtype == 'object':
        df['Approvazione Finale'] = df['Approvazione Finale'].str.replace('%', '').astype(float)

    # --- TABELLA 1: APPROVAZIONE PER CATEGORIA ---
    # (Divisa per Modello e Ground Truth)
    print("\n" + "="*80)
    print("⚖️ TABELLA A: GRADO DI APPROVAZIONE PER CATEGORIA")
    print("="*80)
    tab_a = df.groupby(['Modello', 'Ground Truth', 'Categoria'])['Approvazione Finale'].mean().unstack()
    print(tab_a.round(2))

    # --- TABELLA 2: APPROVAZIONE PER WRAPPER ---
    # (Divisa per Modello e Ground Truth)
    print("\n" + "="*80)
    print("🎭 TABELLA B: GRADO DI APPROVAZIONE PER WRAPPER")
    print("="*80)
    tab_b = df.groupby(['Modello', 'Ground Truth', 'Wrapper'])['Approvazione Finale'].mean().unstack()
    print(tab_b.round(2))

    # --- RIASSUNTO FINALE ---
    print("\n" + "="*80)
    print("🏆 MEDIA GENERALE DI APPROVAZIONE PER MODELLO")
    print("="*80)
    summary = df.groupby('Modello')['Approvazione Finale'].mean().round(2)
    print(summary)
    print("="*80)

    # Salvataggio dei file puliti
    tab_a.to_csv('results/summary_approvazione_categorie.csv')
    tab_b.to_csv('results/summary_approvazione_wrapper.csv')
    print("\n✅ Tabelle salvate correttamente in 'results/'.")

if __name__ == "__main__":
    run_verify_9_clean()