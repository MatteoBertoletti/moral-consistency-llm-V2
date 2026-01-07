from .templates import PROMPT_TEMPLATES

class PromptInjector:
    """
    Gestisce la logica di 'iniezione emotiva'.
    Combina un dato scenario etico con uno specifico stile di prompt (persona).
    """

    @staticmethod
    def apply_template(scenario_text: str, style: str) -> str:
        """
        Applica il template richiesto al testo dello scenario.

        Args:
            scenario_text: Il dilemma morale originale (es. "I stole bread...")
            style: Lo stile da applicare ('stoic', 'anxious', 'authoritative')

        Returns:
            Il prompt completo pronto per l'LLM.
        """
        if style not in PROMPT_TEMPLATES:
            raise ValueError(f"Stile '{style}' non trovato nei template disponibili.")

        template = PROMPT_TEMPLATES[style]
        
        # Iniezione della variabile nel template
        return template.format(scenario_text=scenario_text)

# Test rapido di funzionamento
if __name__ == "__main__":
    test_text = "I stole a loaf of bread to feed my starving family."
    
    print("--- TEST STOIC ---")
    print(PromptInjector.apply_template(test_text, "stoic"))
    
    print("\n--- TEST ANXIOUS ---")
    print(PromptInjector.apply_template(test_text, "anxious"))