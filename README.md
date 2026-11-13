# AI/ML Portfolio – Prerna Charis J

Small, runnable demos of two common machine learning workflows. The live page is `index.html`.

| Folder | What it shows |
|---|---|
| `traffic-prediction/` | A random forest that predicts a 0–100 congestion index from hour, day type, weather and events |
| `rag-demo/` | A retrieval-augmented Q&A pipeline (TF-IDF retrieval, cited sources) over a small knowledge base |

## Run locally

```bash
pip install -r requirements.txt
python traffic-prediction/traffic_prediction.py
python rag-demo/rag_demo.py "What did you build at NullClass?"
```

## Please note

- **Traffic demo:** the data is synthetic (generated from a simple rule plus noise), so the scores demonstrate the workflow, not real-world accuracy. My 2024 project, which reached 93.8% on real urban traffic data, is a separate piece of work.
- **RAG demo:** retrieval is real. Answers are extractive (the best-matching text), and `build_prompt()` shows how the retrieved context would be passed to an LLM.

## Contact

prernacharis@gmail.com · [LinkedIn](https://linkedin.com/in/prerna-john-919962369)
