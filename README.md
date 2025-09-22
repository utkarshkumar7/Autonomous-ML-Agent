# Autonomous ML Agent 🤖

An end-to-end autonomous machine learning agent that ingests datasets, evaluates multiple models, and recommends the best predictor based on evaluation metrics.  
This tool is designed for developers who want a quick, automated way to test and compare ML models without writing boilerplate code.

## 🎥 Demo Video
[![YouTube Video](https://img.shields.io/badge/Watch%20on-YouTube-red?logo=youtube&logoColor=white)](https://www.youtube.com/watch?v=JILLi6H2yWg)
[![Watch the demo](https://img.youtube.com/vi/JILLi6H2yWg/maxresdefault.jpg)](https://www.youtube.com/watch?v=JILLi6H2yWg)

---

## 🚀 Features
- Automated **data preprocessing** (handling missing values, scaling, encoding).
- Supports multiple **machine learning models** out of the box.
- **Evaluation and ranking** of models using standard metrics (accuracy, F1-score, RMSE, etc.).
- **Visualizations** for performance comparison.
- Modular design → easily extendable with new models or metrics.

---

## 🛠️ Tech Stack
- **Python 3.10+**
- **scikit-learn** – training & evaluation
- **pandas / numpy** – data handling
- **matplotlib** – visualization
- **OpenRouter** - used to run LLM prompts and generate python scripts
- **E2B sandbox** - execute generated python scripts


---
## ⚡ Quick Start

Follow these steps to set up and run the Autonomous ML Agent locally:

1. **Clone the repository**
 ```bash
git clone https://github.com/utkarshkumar7/Autonomous-ML-Agent.git
cd Autonomous-ML-Agent
 ```
2. **Create a virtual environment**
  ```bash
  python -m venv venv
  source venv/bin/activate   # On macOS/Linux
  venv\Scripts\activate      # On Windows
  ```
3. **Install required dependencies**
 ```bash
 pip install -r requirements.txt
 ```
4. **Add your OpenRouter + E2B API keys in .env file (private)**
```bash
"OPENROUTER_API_KEY" = #ADD YOUR API KEY
"E2B_API_KEY" = #ADD YOUR API KEY
```
5. **Run the application locally!**
```bash
# It will run on http://localhost:8501 by default
streamlit run main.py
```

---

## 🧭 Architecture & Process Flow

Below is the end-to-end flow for the Autonomous ML Agent. It uses an LLM (e.g., **DeepSeek** via **OpenRouter**) to generate Python code, executes it inside **two isolated E2B sandboxes**, and then produces both a **metrics dataframe** and a **natural-language summary** with a **model leaderboard**.

### Flow Diagram

## 🧭 Architecture & Process Flow

```mermaid
flowchart TD
  U[User / CLI]
  REG[Agent Orchestrator]
  OR[OpenRouter LLM]
  GEN[LLM Code Generation]
  SB1[E2B Sandbox 1 - Data Cleaning]
  SB2[E2B Sandbox 2 - Modeling and Evaluation]
  ART[Artifacts: cleaned.csv, model_results_df, charts]
  SUM[LLM Summarizer]
  VIZ[Leaderboard and Report]

  U -->|dataset.csv + prediction column selection on UI| REG
  REG -->|called by llm_client.py with prompts.py input| OR
  OR --> GEN
  GEN -->|script JSON unpacked as .py script| REG

  REG -->|run cleaning script| SB1
  SB1 -->|cleaned.csv + logs| ART

  REG -->|run training script with cleaned_data.csv| SB2
  SB2 -->|model results dataframe as JSON object| ART

  REG -->|model_results_df| SUM
  SUM -->|natural language summary + insights + leaderboard| VIZ

  ART --> VIZ
