# Early Misinformation Detection

A simple fake news detection system built using Python. You can give it any text or claim, and it will predict whether it is **real or fake**, along with a confidence score.

This project was developed as part of our college work at IIT Ropar. The main goal was to understand how different techniques can be combined to detect misinformation, instead of relying on just one model.

One important part of this project is that everything is implemented from scratch using NumPy, without using deep learning frameworks like PyTorch or TensorFlow.

---

## How it works

The model does not depend on a single method. Instead, it combines multiple ideas to analyze the input text from different angles.

First, the input is checked against a small knowledge base using a retrieval-based approach (RAG). If the system finds strong similarity with known data, it makes a quick prediction.

If the match is not strong enough, the text is processed further using different feature extractors:
- Pattern-based features (similar to CNN)
- Sequence-based features (similar to RNN)
- Structure-based features (similar to RVNN)

All these features are combined and passed into a final classifier, which gives the prediction.

This way, the system tries to be both efficient and reasonably accurate.

---

## How to run

Clone the repository and install dependencies:

```
pip install -r requirements.txt
```

Train the model:

```
python src/train.py
```

Run prediction:

```
python src/predict.py --text "your claim here"
```

Evaluate the model:

```
python src/evaluate.py
```

---

## Techniques used

- TF-IDF :
  --> Used to convert text into numerical form.

- Cosine Similarity : 
  --> Helps in matching input text with the knowledge base.

- RAG (Retrieval-Augmented Generation) :  
  --> Used for quick lookup-based predictions.

- CNN :
  --> Capture local patterns like sensational words.

- RNN :
  --> Capture sequence and flow of text.

- RVNN :  
  --> Capture structure and syntactic signals.

- MLP Classifier :
  --> Final layer that decides whether the text is real or fake.

---

## Project structure

main/
    src/
          nn.py
          model_v3_lazy.py
          feature_extractors.py
          rag_module.py
          knowledge_base.py
          dataset.py
          checkpoint.py
   train.py
   predict.py
   evaluate.py
   .gitignore
   requirements.txt
  

    
---

## Output

The model returns:-
-> Prediction (Real / Fake)
-> Probability score
-> Basic routing info (whether it used retrieval or full model)

---

## Notes

-> This project is mainly for learning and experimentation.
-> The knowledge base is limited, so results may vary on unseen topics.
-. The architecture can be improved further with better data and tuning.

---

## Team Members

- Parth Singhal  
- Ghanisht Kaushal  
- Rahul  
- Amarveer Singh  
- Angad Singh  
