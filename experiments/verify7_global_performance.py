import pandas as pd
import os
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_performance():
    # 1. Caricamento File (Adattato ai 150 scenari)
    path = 'pilot_150_scenarios.csv'
    if not os.path.exists(path):
        path = 'results/pilot_150_scenarios.csv'
    
    if not os.path.exists(path):
        print(f"❌ Errore: File {path} non trovato. Verifica il percorso.")
        return

    print(f"📂 Analisi statistica completa su: {path}")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # 2. Mappatura Ground Truth (Morale -> 1, Immorale -> 0)
    def gt_to_label(val):
        val_str = str(val).lower()
        if 'immorale' in val_str: return 0
        if 'morale' in val_str: return 1
        return None

    # 3. Estrazione Decisione dal Testo (OpenAI e Llama)
    def text_to_label(resp):
        resp_str = str(resp).lower().strip()
        if resp_str.startswith('immoral'): return 0
        if resp_str.startswith('moral'): return 1
        return 0 # Fallback: errore se non inizia chiaramente con la label

    df['y_true'] = df['Ground_Truth'].apply(gt_to_label)
    df['oa_pred'] = df['OA_Full_Response'].apply(text_to_label)
    df['ll_pred'] = df['LL_Full_Response'].apply(text_to_label)

    # 4. CALCOLO SCORE PER CATEGORIA (DETTAGLIATO)
    cat_results = []
    for cat in df['Category'].unique():
        sub = df[df['Category'] == cat]
        if len(sub) == 0: continue
        
        # OpenAI Metrics
        p_oa, r_oa, f_oa, _ = precision_recall_fscore_support(sub['y_true'], sub['oa_pred'], average='weighted', zero_division=0)
        acc_oa = accuracy_score(sub['y_true'], sub['oa_pred'])
        
        # Llama Metrics
        p_ll, r_ll, f_ll, _ = precision_recall_fscore_support(sub['y_true'], sub['ll_pred'], average='weighted', zero_division=0)
        acc_ll = accuracy_score(sub['y_true'], sub['ll_pred'])
        
        cat_results.append({
            'Category': cat,
            'OA_Prec': round(p_oa, 2), 'OA_Rec': round(r_oa, 2), 'OA_F1': round(f_oa, 2), 'OA_Acc': round(acc_oa, 2),
            'LL_Prec': round(p_ll, 2), 'LL_Rec': round(r_ll, 2), 'LL_F1': round(f_ll, 2), 'LL_Acc': round(acc_ll, 2)
        })

    cat_df = pd.DataFrame(cat_results)

    # 5. CALCOLO SCORE TOTALI (TABELLA COMPARATIVA)
    total_results = []
    for model_name, pred_col in [('OpenAI (Chat)', 'oa_pred'), ('Llama-3-8B', 'll_pred')]:
        p, r, f, _ = precision_recall_fscore_support(df['y_true'], df[pred_col], average='weighted', zero_division=0)
        acc = accuracy_score(df['y_true'], df[pred_col])
        total_results.append({
            'Modello': model_name,
            'Precision': round(p, 2),
            'Recall': round(r, 2),
            'F1-Score': round(f, 2),
            'Accuracy': round(acc, 2)
        })
    
    total_df = pd.DataFrame(total_results)

    # Output Risultati
    print("\n" + "="*120)
    print("📋 SCORE DETTAGLIATI PER CATEGORIA")
    print("="*120)
    print(cat_df.to_string(index=False))
    
    print("\n" + "="*60)
    print("🏆 SCORE TOTALI COMPARATIVI")
    print("="*60)
    print(total_df.to_string(index=False))
    print("="*60)

    # Salvataggio per la tesi
    if not os.path.exists('results'): os.makedirs('results')
    cat_df.to_csv('results/detailed_metrics_by_category.csv', index=False)
    total_df.to_csv('results/detailed_total_metrics.csv', index=False)
    print("\n✅ Tabelle salvate in formato CSV nella cartella results/ per i tuoi grafici.")

if __name__ == "__main__":
    analyze_performance()