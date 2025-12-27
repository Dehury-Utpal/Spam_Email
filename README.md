

## 📌 Project Overview
This project performs **sentiment analysis** on the **IMDB movie reviews dataset** using a **Long Short-Term Memory (LSTM) neural network**.  
The goal is to classify reviews as **positive** or **negative** based on the text content.  

**Why LSTM?**  
LSTMs are a type of Recurrent Neural Network (RNN) capable of learning long-term dependencies in sequential data. They are highly effective for Natural Language Processing (NLP) tasks like sentiment analysis because they can capture the context and sequential relationships in text.

---

## 🛠️ Technologies Used
- **Python 3.x**  
- **TensorFlow / Keras** – building and training LSTM models  
- **NumPy, Pandas** – data manipulation and preprocessing  
- **Matplotlib, Seaborn** – visualization of training metrics and results  
- **NLTK** – text tokenization and preprocessing  
- **Google Colab** – for running notebooks with free GPU acceleration  

---

## 📂 Dataset
- **IMDB Large Movie Review Dataset**  
- Contains 50,000 labeled reviews (25,000 for training, 25,000 for testing)  
- Binary classification: Positive / Negative  
- Dataset source: [IMDB Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)  

---

## 🔹 Preprocessing
- Tokenized the reviews using top **10,000 words**  
- Converted reviews to sequences of integers  
- Padded sequences to **max length 200** for uniform input  
- Removed stopwords and special characters (via NLTK)  
- Split dataset into **training and testing sets**  

---

## 🔹 Model Architecture
- **Embedding Layer**: Converts words into 128-dimensional vectors  
- **LSTM Layer**: 128 units with dropout (0.2) and recurrent dropout (0.2)  
- **Dense Layer**: Sigmoid activation for binary classification  
- **Loss Function**: Binary Cross-Entropy  
- **Optimizer**: Adam  
- **Metrics**: Accuracy  

---

## 🚀 Features
- Training and evaluation of LSTM model  
- Visualization of **training/validation accuracy and loss curves**  
- Predict sentiment of **custom reviews**  
- Saves preprocessing, model structure, and evaluation results  
- Provides clear results for positive and negative classification  

---

## 📊 Results
- **Training Accuracy:** ~88%  
- **Validation Accuracy:** ~85%  
- **Test Accuracy:** ~85%  
- Loss converges steadily after 10–15 epochs  
- Model correctly classifies both positive and negative reviews

### 📝 Example Predictions
| Review | Predicted Sentiment |
|--------|-------------------|
| "This movie was fantastic! The acting was brilliant." | Positive ✅ |
| "I wasted two hours of my life. Terrible plot." | Negative ❌ |
| "Amazing storyline and characters, loved it." | Positive ✅ |
| "Poor script and bad acting. Do not recommend." | Negative ❌

## 🔮 Future Improvements
- Use **Bidirectional LSTM** or **GRU layers** for improved performance  
- Experiment with **pre-trained embeddings** like Word2Vec or GloVe  
- Hyperparameter tuning (layers, dropout, batch size)  
- Deploy as a **web application** using Flask, Django, or Streamlit  
- Include more datasets for multi-domain sentiment analysis
## 🙏 Acknowledgments
- IMDB dataset from [Stanford IMDB Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)  
- TensorFlow & Keras documentation  
- Inspiration from deep learning tutorials and NLP research  

## 👤 Author
**UTPAL DEHURY**  
🎓 B.Tech CSE @ Silicon University, Bhubaneswar 
