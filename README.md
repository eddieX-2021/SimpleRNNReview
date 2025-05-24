# Anime Review Sentiment Analysis using Simple RNN

This project is a **text classification system** designed to predict sentiment (Positive or Negative) from anime reviews scraped from MyAnimeList (MAL). It utilizes a **Simple Recurrent Neural Network (RNN)** built with TensorFlow/Keras. The front-end interface is powered by **Streamlit** for interactive inference.

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/anime-sentiment-rnn.git
cd anime-sentiment-rnn

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt

pip install tensorflow streamlit scikit-learn pandas numpy


To train the sentiment classification model on MAL reviews:

Load the dataset from the CSV (e.g., reviews.csv)  (dataset is avaiable on kaggle it is not here because it is too big)

Preprocess the review texts and binarize the review scores

Tokenize and pad the sequences

Build a Simple RNN model

Train the model with early stopping

Training is done in the provided notebook: simplernn.ipynb

After training:

Tokenizer saved as mal_tokenizer.pkl

Model saved as anime_review_model.h5


Launch the Streamlit web app:

bash
streamlit run main.py
You'll see an interface to input anime reviews and get predicted sentiment with confidence level.


📌 Project Purpose
This project demonstrates:

End-to-end NLP processing pipeline

Simple RNN implementation on real-world data

Model deployment with Streamlit

⚠️ Limitations
Current challenges:

Noisy/lengthy review texts

Imbalanced labels (mostly high scores)

Simple RNN limitations for long sequences

🚀 Future Improvements
Potential enhancements:

Upgrade to LSTM/GRU or Transformers

Use pretrained embeddings (GloVe, BERT)

Better text cleaning (remove HTML/formatting)

Dataset balancing techniques

Multi-class sentiment analysis

🙋‍♂️ Author
Eddie Xiao
Computer Science and Economics Student
University of Virginia
Summer ML/NLP Learning Project