# Word2Vec from Scratch: Skip-gram & CBOW Implementation 🔍

This project implements two popular neural network architectures for learning word embeddings — **Skip-gram** and **Continuous Bag of Words (CBOW)** — from scratch using **PyTorch**, without relying on pre-trained models or external libraries like Gensim.

The goal is to deeply understand how word embeddings are generated, visualize their semantic structure, and validate the learning through analogy reasoning tasks.

---

## 🔧 Technologies & Tools
- **Python**
- **PyTorch**
- **NLTK** (tokenization, stopwords)
- **Matplotlib** (visualization)
- **Scikit-learn** (t-SNE for dimensionality reduction)
- **Jupyter Notebook**

---

## 📌 Project Features

- **Data Preprocessing**:
  - Custom text corpus cleaned using Python and NLTK
  - Tokenization, stopword removal, vocabulary generation
  - One-hot encoding and word-to-index mapping

- **Skip-gram Architecture**:
  - Center word predicts surrounding context
  - `nn.Embedding` used to learn low-dimensional word vectors
  - Trained with CrossEntropyLoss + Adam optimizer

- **CBOW Architecture**:
  - Context words used to predict center word
  - Averaged embeddings fed to output layer
  - Separate training loop and loss evaluation

- **Training & Evaluation**:
  - Epoch-wise training loss display
  - Word analogy testing: e.g. `king - man + woman = ?`
  - Embedding matrix saved for reuse

- **Visualization**:
  - t-SNE used to project high-dimensional embeddings to 2D
  - Plotted word clusters to observe semantic proximity

---

## 📊 Project Output

- Learned vector representations (embeddings) for each word in the corpus
- Visual 2D map of word embeddings
- Demonstrated semantic relationships via clustering and analogy tasks
