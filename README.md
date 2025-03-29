
# 📰 BERT Financial News Sentiment Analysis

This project uses a pre-trained BERT model to perform **sentiment analysis** on financial news headlines and articles, helping evaluate whether news content is **bullish**, **bearish**, or **neutral** in tone.

## 🔍 Overview

- Fine-tuned a BERT model (`BertForSequenceClassification`) for sentiment classification.
- Used real-world financial news datasets to train and test the model.
- Aimed at understanding market sentiment to support investment decisions and financial analysis.

## 🧠 Key Concepts

- **NLP (Natural Language Processing)**: Applied tokenization, attention masks, and classification techniques.
- **Transfer Learning**: Leveraged pre-trained BERT for improved performance on domain-specific financial text.
- **Evaluation Metrics**: Measured accuracy, precision, recall, and F1 score using `sklearn`.

## 📁 Dataset

- Downloaded from Inspirit AI’s financial news project:
  - `finance_train.csv`
  - `finance_test.csv`

Each row includes:
- A news headline or snippet
- Its associated sentiment label (0 = negative, 1 = neutral, 2 = positive)

## 🧪 Tech Stack

- Python, PyTorch, Hugging Face Transformers, scikit-learn, Pandas, Matplotlib

## 🧾 Sample Code Snippet

```python
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels = 3,
    output_attentions = False,
    output_hidden_states = False,
)
```

## 📊 Sample Output

Achieved ~85% accuracy on validation data using fine-tuned BERT. Sentiment predictions mapped financial text to:
- 📉 Negative
- 📈 Positive
- ⚖️ Neutral

## 🧰 To Run

```bash
pip install transformers torch pandas scikit-learn
```

Then execute the script in order, starting with data download and preprocessing.
