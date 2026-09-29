# AutoSenti - AI-Based Automotive Review & Customer Sentiment Analytics

> Aspect-Based Sentiment Analysis (ABSA) for automotive customer reviews,  
> powered by BERT and XLM-RoBERTa transformers.

---

## Features

| Feature | Description |
|---------|-------------|
| **Vehicle review analysis** | Enter a review → detect aspects → classify sentiment per aspect |
| **Multi-aspect support** | One review can yield multiple aspect-sentiment pairs |
| **Confidence scores** | Softmax probability for each prediction |
| **Confidence overview** | Compare confidence across detected aspects |
| **Dark UI** | Focused Streamlit interface for automotive reviews |

---

## Supported Automotive Aspects

Vehicle · Engine · Battery · Mileage · Safety · Comfort · Service · Infotainment · Price

Configurable in `config.yaml` — add any aspect without touching code.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the application

```bash
python run.py
# or
streamlit run app/main.py
```

### 3. Open in browser

```
http://localhost:8501
```

### 4. Use the app

1. Select **BERT** or **XLM-RoBERTa** in the sidebar → click **⚡ Load Model**
2. Paste a vehicle review into the input box → click **Analyze review**
3. Review the detected aspects, sentiment labels, and confidence scores

---

## Project Structure

```
├── app/
│   ├── main.py                  # Streamlit entry point + global CSS
│   ├── ui_components.py         # Reusable result components
│   └── pages/page_home.py       # Vehicle review analysis page
├── src/
│   ├── config_loader.py         # Centralised YAML config access
│   ├── data/
│   │   ├── preprocessor.py      # Text preprocessing pipeline
│   │   └── validator.py         # Review input validation
│   ├── models/
│   │   ├── absa_dataset.py      # Sentiment label mappings
│   │   └── model_factory.py     # Tokenizer/model loader + device selection
│   ├── inference/
│   │   ├── aspect_extractor.py  # Keyword-based aspect extraction
│   │   ├── sentiment_classifier.py  # Transformer sentiment inference
│   │   └── inference_engine.py  # Orchestrator (single + batch)
│   └── analytics/visualizer.py  # Confidence chart rendering
├── config.yaml                  # Centralized configuration
├── requirements.txt
└── run.py                       # Convenience launcher
```

## Models

| Model | HuggingFace ID | Best For |
|-------|---------------|---------|
| BERT | `bert-base-uncased` | English reviews, fast |
| XLM-RoBERTa | `xlm-roberta-base` | Multilingual, better generalisation |

**Encoding:** Sentence-pair format — aspect as Sentence A, review as Sentence B.  
This gives the model aspect-conditioned context during classification.

---

## Hardware Requirements

| | Minimum | Recommended |
|-|---------|-------------|
| CPU | Modern multi-core | — |
| RAM | 8 GB | 16 GB |
| GPU | Not required (inference) | NVIDIA CUDA GPU (training) |

---

## Configuration

All system parameters are in [`config.yaml`](config.yaml):

- **Aspect taxonomy** — add/remove aspects and keywords
- **Model settings** — name, max token length

---

## Constraints (as per SRS)

- Aspect extraction is keyword-based; **implicit aspects are not detected**.
- Sarcasm and figurative language may reduce accuracy.
- Confidence scores are model probability estimates, not guaranteed correctness.
- The system is an **analytical tool**, not a substitute for human decisions.

---

## License

This project is for academic and research purposes.
