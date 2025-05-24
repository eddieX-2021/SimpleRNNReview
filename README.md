# Anime Review Sentiment Analysis using Simple RNN 🎭

This project is a text classification system designed to predict sentiment (Positive or Negative) from anime reviews scraped from MyAnimeList (MAL). It utilizes a Simple Recurrent Neural Network (RNN) built with TensorFlow/Keras, with a Streamlit-powered front-end interface.

![Sentiment Analysis](https://via.placeholder.com/800x400?text=Anime+Sentiment+Analysis+Demo) *(placeholder for project screenshot)*

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- pip

### Installation
```bash
# Clone the repository
git clone https://github.com/your-username/anime-sentiment-rnn.git
cd anime-sentiment-rnn

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt


## 🧠 Model Training

To train the sentiment classification model on MAL reviews:

1. Load the dataset from `reviews.csv` (available on Kaggle)
2. Preprocess the review texts and binarize the review scores
3. Tokenize and pad the sequences
4. Build a Simple RNN model
5. Train the model with early stopping

Training is done in the provided notebook: `simplernn.ipynb`

After training:
- Tokenizer saved as `mal_tokenizer.pkl`
- Model saved as `anime_review_model.h5`

## 💻 Running the Web App

Launch the Streamlit web app:
```bash
streamlit run main.py


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