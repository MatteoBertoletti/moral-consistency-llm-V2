import pandas as pd
from typing import List
import uuid
from .schemas import MoralScenario

class DataLoader:
    @staticmethod
    def load_ethics_commonsense(split="train", limit=5) -> List[MoralScenario]:
        print(f"🔄 Tentativo di scaricamento ETHICS (commonsense)...")
        
        base_url = "https://huggingface.co/datasets/hendrycks/ethics/resolve/main/data/commonsense"
        url = f"{base_url}/{split}.csv"
        
        try:
            # Leggiamo SENZA nomi colonne per evitare confusione
            df = pd.read_csv(url, header=None)
        except Exception as e:
            raise RuntimeError(f"❌ Errore download: {e}")

        # --- AUTO-RILEVAMENTO COLONNE ---
        # Capiamo quale colonna è il testo (quella con le stringhe più lunghe)
        col_0_len = df[0].astype(str).str.len().mean()
        col_1_len = df[1].astype(str).str.len().mean()

        if col_0_len > col_1_len:
            text_col_idx, label_col_idx = 0, 1
            print("🧠 Rilevato: Colonna 0 è il TESTO, Colonna 1 è la LABEL")
        else:
            text_col_idx, label_col_idx = 1, 0
            print("🧠 Rilevato: Colonna 1 è il TESTO, Colonna 0 è la LABEL")

        scenarios = []
        skipped_count = 0
        
        for i, row in df.iterrows():
            if len(scenarios) >= limit:
                break
            
            # Usiamo gli indici rilevati dinamicamente
            text_content = str(row[text_col_idx])
            raw_label = row[label_col_idx]
            
            # --- FILTRI DI PULIZIA ---
            # 1. Filtro lunghezza
            if len(text_content.strip()) < 10:
                skipped_count += 1
                # Debug: vediamo cosa stiamo scartando (solo i primi 3 errori)
                if skipped_count <= 3:
                    print(f"⚠️ SCARTATO (Troppo corto): '{text_content}'")
                continue
                
            # 2. Filtro parola "edited"
            if "edited" in text_content.lower():
                skipped_count += 1
                if skipped_count <= 3:
                     print(f"⚠️ SCARTATO (Contiene 'edited'): '{text_content}'")
                continue
            
            # Normalizzazione Label (Accetta sia 0/1 che stringhe)
            try:
                is_acceptable = int(raw_label) == 1
            except:
                is_acceptable = False # Fallback

            label_str = "acceptable" if is_acceptable else "unacceptable"
            
            scenario = MoralScenario(
                id=f"ethics_cm_{i}",
                text=text_content,
                source_dataset="ethics",
                label=label_str
            )
            scenarios.append(scenario)
            
        print(f"✅ Caricati {len(scenarios)} scenari PULITI (Saltati {skipped_count} sporchi).")
        return scenarios

if __name__ == "__main__":
    # TEST IMMEDIATO
    try:
        data = DataLoader.load_ethics_commonsense(limit=5)
        print("\n" + "="*50)
        print("VERIFICA QUALITÀ DATI")
        print("="*50)
        
        if len(data) > 0:
            print(f"🔹 ESEMPIO REALE: \"{data[0].text}\"")
            print(f"🏷️ LABEL: {data[0].label}")
            print("\n🟢 TEST SUPERATO: I dati sono corretti.")
        else:
            print("🔴 TEST FALLITO: Ancora 0 dati.")
            
    except Exception as e:
        print(f"🔴 ERRORE: {e}")