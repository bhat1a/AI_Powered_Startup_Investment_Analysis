---
title: VentureAI
emoji: 🚀
colorFrom: purple
colorTo: indigo
sdk: streamlit
sdk_version: 1.56.0
app_file: main_app.py
pinned: false
---

# VentureAI — AI-Powered Startup Investment Analysis

A Streamlit app with two modules:

- **Startup Investment Advisor** — LightGBM model (84% accuracy, 90.5% CV F1) with SHAP explanations and Cerebras LLM narrative
- **CSV Data Visualiser** — LangGraph + Cerebras pipeline that generates interactive Highcharts from natural language

Built as a final year project at MITADT University.

## Setup

Set the following secrets in HF Space settings:

| Key | Value |
|---|---|
| `API_KEY` | Your Cerebras API key |
| `MODEL` | `qwen-3-235b-a22b-instruct-2507` |
| `INTENT_MODEL` | `llama3.1-8b` |
