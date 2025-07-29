# Stock Market Sentiment Analysis Using Twitter Data

## 1. Introduction

Understanding how people feel about the stock market is relevant for several groups (including investors) due to its influence on market behavior. Social media platforms like Twitter provide free and real-time access to opinions on market trends.

This project focuses on classifying tweets into three sentiment categories:  
- **Bullish** (expecting prices to go up)  
- **Bearish** (expecting prices to go down)  
- **Neutral**

The goal is to create a Natural Language Processing (NLP) model that predicts the sentiment behind each tweet, enabling real-time extraction of market sentiment to inform decision-making.

The main challenge lies in capturing the varied and informal expressions used on social media, including abbreviations and unique slang, which complicates natural language understanding.
To address this, the project explores:

- Data exploration to understand tweet content and sentiment distribution  
- Text preprocessing for model readiness  
- Building and testing various classification models (both pretrained and non-pretrained)  
- Evaluation and comparison of model performance  

The objective is a robust model that works well technically and adapts to the noisy, real-world nature of Twitter data.

---

## 2. Data Exploration

The training dataset contains **9,543** tweets with two columns: text and sentiment label distributed as follows:

| Label     | Count | Percentage (%) |
|-----------|--------|----------------|
| Bearish (0) | 1,442  | 15.11%         |
| Bullish (1) | 1,923  | 20.15%         |
| Neutral (2) | 6,178  | 64.74%         |

- Common stop words dominated the vocabulary, alongside non-informative tokens.
- Word clouds (excluding stop words) highlighted more meaningful terms.
- Separate word clouds per sentiment class revealed distinct vocabulary after removing high-frequency non-informative tokens.
- Hashtag usage was analyzed both in total and average per tweet, adjusting for class imbalance.

---

## 3. Data Preprocessing

- Checked for missing values and duplicates — none found.
- Created two new variables:  
  - `text_length`: number of characters per tweet  
  - `word_count`: number of words per tweet
- Distribution of text lengths was approximately normal, with higher variability in Neutral tweets.
- Tweets under the 0.5th percentile in length were removed as outliers due to low information content.

### 3.1 Language Detection and Translation

- Tweets were written in multiple languages.
- **FastText (lid.176.bin)** from Facebook AI was used for language detection, optimized for short texts and supporting 176 languages.
- Non-English tweets were translated to English using **MarianMT models** from the Helsinki-NLP collection for high-quality neural machine translation.
- Manual checks confirmed translation quality.

### 3.2 Text Cleaning

Applied the following to prepare text for non-pretrained models:

- Lowercasing  
- Removal of URLs, numbers, punctuation, and non-alphabetic characters  
- Removal of English stopwords  
- Optional spelling correction, stemming, and lemmatization  
- Removal of extremely rare words to reduce noise

The dataset was then split into train/validation (80/20) using stratified sampling on both cleaned and raw texts (for pretrained models).

---

## 4. Feature Engineering

Text was transformed into numerical formats for model input:

- **Classical methods:** Bag of Words (BoW), TF-IDF  
- **Semantic embeddings:** Word2Vec, GloVe, RoBERTa, BERTweet, Universal Sentence Encoder

---

## 5. Classification Models

Tested multiple classifiers with different embeddings:

- **5.1 Naive Bayes**  
- **5.2 Logistic Regression**  
- **5.3 k-Nearest Neighbors (kNN)**  
- **5.4 Multilayer Perceptron (MLP)**  
- **5.5 Long Short-Term Memory (LSTM)**  
- **5.6 XGBoost**  
- **5.7 Ensemble Model (Voting Classifier)**  
- **5.8 RoBERTa (Hugging Face pipeline)**  
- **5.9 BERTweet**  
- **5.10 BART**

---

## 6. Evaluation and Results

- Created heatmaps summarizing **accuracy, F1 score, precision, and recall** across model-embedding combinations.
- Detailed F1 score breakdowns showed the relative effectiveness of embeddings per model and vice versa.

### Key Findings:

- **Bag of Words (BoW)** performed well, especially with Naive Bayes, except in precision where TF-IDF excelled.
- **Word2Vec** consistently underperformed, often below 40% in several models.
- **GloVe** had moderate success, particularly with LSTM (scores > 65%).
- **Transformer-based embeddings (RoBERTa, BERTweet)** showed the best and most consistent performance.
  - RoBERTa was the top performer across nearly all models.
  - BERTweet closely followed and was a strong alternative.
- **BART** performed poorly in accuracy and other metrics.
- The best overall model was the **Ensemble with RoBERTa embeddings**, achieving:  
  - **Accuracy: 84%**  
  - **F1, Precision, Recall:** between 0.77 and 0.79

### Ensemble Details:

- Combines Logistic Regression, MLP, and XGBoost using soft voting.
- GridSearchCV optimized voting strategy and weights — best with weights [1, 1, 2], favoring XGBoost.
- Incorporated **SMOTE** for balancing classes, improving generalization.
- Final model metrics:  
  - Validation macro F1-score: 0.7958  
  - Training macro F1-score: 0.9924  
- Reduced overfitting and improved generalization compared to untuned version.

---

## 7. Conclusion

This project successfully classified stock market sentiment in tweets using traditional and transformer-based NLP techniques. Highlights include:

- Evaluation of multiple feature representations (TF-IDF, GloVe, FastText, RoBERTa, BERTweet, USE)  
- Exploration of both non-pretrained and pretrained classification models  
- Implementation of advanced methods beyond course scope, including additional transformer embeddings and models (BERTweet, BART)  
- Transformer models (RoBERTa, BERTweet) outperformed simpler methods, though basic models still provided solid baselines  
- Best results achieved with an ensemble using RoBERTa embeddings (F1-score ~78%)
