# AI/ML Portfolio – Prerna Charis J

Live portfolio: https://prernacharis-ai.github.io/ai-ml-portfolio/

Small, runnable demos of two common machine learning workflows.

| File | What it shows |
|---|---|
| traffic_prediction.py | A random forest that predicts a 0–100 congestion index from hour, day type, weather and events |
| rag_demo.py, knowledge_base.json | A retrieval-augmented Q&A pipeline (TF-IDF retrieval, cited sources) over a small knowledge base |

## Run locally

pip install -r requirements.txt
python traffic_prediction.py
python rag_demo.py "What did you build at NullClass?"

## Please note

- Traffic demo: the data is synthetic (generated from a simple rule plus noise), so the scores demonstrate the workflow, not real-world accuracy. My 2024 project, which reached 93.8% on real urban traffic data, is separate work.
- RAG demo: retrieval is real. Answers are extractive (the best-matching text), and build_prompt() shows how the retrieved context would be passed to an LLM.

## Contact

prernacharis@gmail.com
