# Liquid Morality: Investigating Ethical Alignment Fragility in LLMs

This repository contains the official implementation for the study on **Liquid Morality** in Large Language Models (LLMs). The research investigates how personality wrappers system prompts such as Stoic, Anxious, and Authoritative—deform the ethical judgment and internal consistency of OpenAI and Llama-3-8B.

---

## 🚀 Key Scientific Findings

* **Justice Inversion & Inconsistency**: Llama-3-8B demonstrates a significant 40.0% average inconsistency rate across personas, while OpenAI remains more stable at 10.0%.
* **Justice Volatility**: The Justice category exhibits the highest fragility, with a 50.0% discrepancy in moral verdicts solely due to persona shifts.
* **Categorical Blindness (OpenAI)**: OpenAI exhibits "Algorithmic Rigorism," registering 0.0% accuracy in detecting immoral deontological violations and 0.0% accuracy in identifying moral virtuous acts.
* **Global Performance**: Llama-3-8B achieves a global F1-score of 0.49, slightly outperforming OpenAI's 0.46 under personality-induced pressure.
* **Accuracy Peak**: Both models perform best in the Commonsense category, with OpenAI reaching 80.0% and Llama-3-8B 70.0% accuracy.
* **Token Sensitivity**: Logprob-based masking analysis identified key "trigger tokens" (e.g., 'anxious', 'authoritative', 'required') that shift moral conviction scores by up to 30.0%.

---

## 📊 Methodology: NLU Pipeline

The evaluation utilizes a multi-stage **Natural Language Understanding (NLU)** pipeline to quantify model conviction beyond binary labels:

1.  **Sentiment Analysis (BERT)**: Evaluates the emotional valence of the model's response.
2.  **Stance Detection (BART-NLI)**: Measures the logical accord between the argument and the ethical prompt.
3.  **Final Approval Index**:
    > **Approval = (Sentiment × 0.4) + (Stance × 0.6)**

---

## 💻 Execution Guide

To replicate the results presented in the study, follow this specific order:

### 1. Data Generation (The Runner)
First, execute the pipeline runner to process ETHICS scenarios through the LLMs. This populates the `results/` directory with raw NLU data. After you have to execute in order all the verify files in the experiments folder.

```bash
python pipeline/runner.py