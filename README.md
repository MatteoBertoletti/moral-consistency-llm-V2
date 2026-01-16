# Liquid Morality: Investigating Ethical Alignment Fragility in LLMs

This repository contains the official implementation for the study on "Liquid Morality" in Large Language Models (LLMs). The research investigates how personality wrappers—system prompts such as Stoic, Anxious, and Authoritative—deform the ethical judgment and internal consistency of OpenAI and Llama-3-8B.

## 🚀 Key Scientific Findings

* **Justice Inversion & Inconsistency**: Llama-3-8B demonstrates a significant 40.0% average inconsistency rate across personas, while OpenAI remains more stable at 10.0%.
* **Justice Volatility**: The Justice category exhibits the highest fragility, with a 50.0% discrepancy in moral verdicts solely due to persona shifts.
* **Categorical Blindness (OpenAI)**: OpenAI exhibits "Algorithmic Rigorism," registering 0.0% accuracy in detecting immoral deontological violations and 0.0% accuracy in identifying moral virtuous acts.
* **Global Performance**: Llama-3-8B achieves a global F1-score of 0.49, slightly outperforming OpenAI's 0.46 under personality-induced pressure.
* **Accuracy Peak**: Both models perform best in the Commonsense category, with OpenAI reaching 80.0% and Llama-3-8B 70.0% accuracy.
* **Token Sensitivity**: Logprob-based masking analysis identified key "trigger tokens" (e.g., 'anxious', 'authoritative', 'required') that shift moral conviction scores by up to 30.0%.

## 🛠️ Repository Structure

```text
├── core/                # Core logic (Models, Evaluators, Prompts)
├── data/                # Dataset loaders and ETHICS subset
├── pipeline/            # Main execution logic
│   ├── runner.py        # PRIMARY ENTRY POINT: Data generation
│   └── word_attribution.py # Masking and token importance analysis
├── experiments/         # Evaluation and metrics (Verify scripts)
│   ├── verify1_setup_check.py
│   ├── verify2_data_inspection.py
│   ├── verify3_asymmetry_analysis.py
│   ├── verify4_robustness_consistency.py
│   ├── verify5_categorical_benchmarks.py
│   ├── verify6_logprob_masking.py
│   ├── verify7_global_performance.py
│   ├── verify8_nlu_sentiment_stance.py
│   └── verify9_approval_index.py
├── results/             # Directory for output logs, CSVs, and plots
└── requirements.txt     # Python dependencies
```

## 📊 Methodology: NLU Pipeline

The evaluation utilizes a multi-stage **Natural Language Understanding (NLU)** pipeline to quantify model conviction beyond binary labels:

* **Sentiment Analysis (BERT):** Evaluates the emotional valence of the model's response.
* **Stance Detection (BART-NLI):** Measures the logical accord between the argument and the ethical prompt.
* **Final Approval Index:** A weighted metric calculated as:
    $$Approval = (Sentiment \times 0.4) + (Stance \times 0.6)$$

---

## 💻 Execution Guide

To replicate the results and metrics presented in the study, files must be executed in the following **specific order**:

### 1. Data Generation (The Runner)
First, you must execute the pipeline runner. This script processes the **ETHICS** scenarios through the selected LLMs under the three personality wrappers. It extracts Sentiment and Stance scores and populates the `results/` directory with raw data.

```bash
python pipeline/runner.py

After you have to execute in order all the verify files in the experiments folder.