import numpy as np

class TargetedInterpretability:
    def __init__(self, model_instance):
        self.model = model_instance
        # Definiamo le parole chiave per ogni wrapper
        self.wrapper_keywords = {
            "anxious": ["immediately", "urgent", "hurry", "anxious", "worried"],
            "authoritative": ["judge", "official", "authority", "strict", "must"],
            "stoic": ["neutral", "objective", "calm", "unbiased", "fact"]
        }

    def get_shap_values(self, system_prompt, full_prompt, wrapper_type, target_label):
        """
        Calcola l'attribuzione (SHAP) solo per le 5 parole chiave del wrapper scelto.
        """
        keywords = self.wrapper_keywords.get(wrapper_type.lower(), [])
        
        # 1. Baseline: Probabilità con il prompt intero
        _, p_data_orig = self.model.generate_with_probs(system_prompt, full_prompt)
        base_prob = self._extract_prob(p_data_orig, target_label)
        
        shap_values = {}

        # 2. Analisi mirata Leave-One-Out (LOO)
        for word in keywords:
            if word.lower() in full_prompt.lower():
                # Creiamo il prompt senza la parola chiave (perturbazione)
                # Usiamo regex per rimuovere solo la parola intera
                perturbed_prompt = full_prompt.replace(word, "[REDACTED]")
                
                _, p_data_new = self.model.generate_with_probs(system_prompt, perturbed_prompt)
                new_prob = self._extract_prob(p_data_new, target_label)
                
                # SHAP value = Differenza di probabilità
                shap_values[word] = round(base_prob - new_prob, 4)
        
        return shap_values

    def _extract_prob(self, p_data, label):
        if not p_data: return 0.0
        prefix = label.lower()[:3]
        for p in p_data:
            token = str(getattr(p, 'token', p.get('token', ''))).strip().lower()
            if token.startswith(prefix):
                return np.exp(getattr(p, 'logprob', -100.0))
        return 0.0