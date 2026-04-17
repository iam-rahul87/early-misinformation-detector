# Early Misinformation Detection

A simple fake news detector built using Python. Give it any text claim and it will tell you if it's real or fake with a confidence score.

This was our college project at IIT Ropar. We built everything from scratch without using PyTorch or TensorFlow.

## How to run

```bash
pip install -r requirements.txt
python train.py
python predict.py --text "your claim here"
```

## What we used

- Python
- NumPy
- scikit-learn


## Model/Techniques used

- TF-IDF --> To convert text into numbers.
- Cosine similarity --> To Match claims against knowledge base.
- RAG --> knowledge base lookup.
- CNN --> Pattern detection.
- RNN --> Sequence features.
- RVNN --> Syntactic structure features.
- MLP classifier --> Final fake/real decision.

## Team Members

- Parth singhal
- Ghanisht kaushal
- Rahul
- Amarveer singh
- Angad singh
  
