<<<<<<< HEAD
# MisinfoNet

We built this for a college project and it turned into something we're actually proud of. It takes any text claim and tells you whether it's likely fake or real — with a confidence score. No pretrained models, no APIs, just NumPy and scikit-learn doing all the heavy lifting.

The focus is on political misinformation — elections, wars, terrorism, that kind of thing. Mostly because that's where fake news actually causes harm.

---

## What's the idea

Most fake news detectors are black boxes. You feed text in, a number comes out, and you have no idea what happened in the middle. We wanted to actually understand what we were building, so we wrote everything from scratch — the neural network, the optimizer, the feature extractors, all of it.

The model has two paths. First it checks if the claim is similar to anything in our knowledge base (500+ entries we wrote by hand). If it finds a match, it uses a small fast classifier and stops there — done in milliseconds. If the claim is something the KB hasn't seen, it runs three different feature extractors that look for things like sensationalist language, narrative structure, hedging phrases, and passive voice. Those features go into a bigger classifier that gives the final verdict.

We call it lazy routing because the heavy computation only runs when it actually needs to.

---

## Running it

```bash
git clone https://github.com/YOUR_USERNAME/misinfonet.git
cd misinfonet
pip install -r requirements.txt
python train.py
```

Training takes maybe 3-4 minutes on a laptop. After that:

```bash
# try a claim
python predict.py --text "NATO deliberately provoked Russia into invading Ukraine"

# or just type claims one by one
python predict.py --interactive

# or run the demo we set up
python predict.py --demo
```

To see how it actually performs on real examples (not just the training data):

```bash
python evaluate.py --real --verbose
```

The `--verbose` flag shows you every single claim it got wrong, which is honestly more useful than just seeing the accuracy number.

---

## Files

```
misinfonet/
├── src/
│   ├── model_v3_lazy.py       routing logic and the two classifiers
│   ├── nn.py                  the neural network we wrote from scratch
│   ├── rag_module.py          knowledge base lookup
│   ├── feature_extractors.py  the three feature extractors
│   ├── dataset.py             generates training data from templates
│   ├── knowledge_base.py      500+ claims we wrote by hand
│   └── checkpoint.py          saving and loading the model
├── train.py
├── predict.py
└── evaluate.py
```

---

## Tweaking the training

```bash
python train.py --epochs 100 --threshold 0.20 --dropout 0.3
```

The threshold controls when the KB path kicks in vs the neural path. Lower threshold means more claims go through the KB path. Higher means more go through neural. 0.30 worked best for us but it depends on your data.

---

## Honest limitations

The training data is synthetic — we generated it from templates, which means the model is really good at catching obvious fake news (the BREAKING EXPOSED SHOCKING stuff) but struggles with subtle disinformation that's written in a calm, neutral tone. That's the harder problem and we haven't fully solved it.

Also everything is in English and the KB only covers political topics. Give it a claim about something else and it'll still try to answer but don't trust it too much.

---

## Who made this

Rahul, Parth singhal, ghanisht kaushal

IIT Ropar — 2026

---

## License

MIT. Do whatever you want with it.
=======
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
  
>>>>>>> 07611f90f9f0093a838c2d3b91640850c4f275fc
