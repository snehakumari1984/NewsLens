# 🔎 NewsLens AI – AI-Powered News Credibility Analysis System

An AI-powered Fake News Detection System that uses Natural Language Processing (NLP) and Machine Learning to analyze news articles and identify linguistic patterns associated with potentially misleading information.

🌐 **Live Demo:** https://newslens-hfysjkhpbctwzd3hwftrj.streamlit.app/

💻 **GitHub Repository:** https://github.com/snehakumari1984/NewsLens

---

## 📌 Project Overview

NewsLens AI is a machine learning-based web application designed to analyze news headlines and article content to predict whether a news article is likely to be real or misleading.

The system uses TF-IDF (Term Frequency–Inverse Document Frequency) for text feature extraction and Logistic Regression for classification.

It provides an interactive dashboard where users can submit news content, view predictions, explore prediction history, and examine model performance.

## ✨ Key Features

- **News Credibility Analysis:** Analyze news articles using a trained ML classifier.
- **NLP-Based Processing:** Extract meaningful textual features using TF-IDF.
- **Machine Learning Prediction:** Classify news using Logistic Regression.
- **Prediction History:** Maintain records of previous analyses using SQLite.
- **Model Performance Dashboard:** Explore model evaluation metrics and comparisons.
- **Sample News Articles:** Test the application using built-in examples.
- **Interactive Interface:** Navigate through a user-friendly Streamlit dashboard.

## 🛠️ Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Machine Learning | Scikit-learn |
| NLP | TF-IDF Vectorizer |
| Classification | Logistic Regression |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly |
| Database | SQLite |
| Web Framework | Streamlit |
| Model Storage | Joblib / Pickle |

## 🧠 Machine Learning Workflow

1. **Data Collection:** Use a labeled news dataset for model development.
2. **Text Preprocessing:** Clean and prepare textual data.
3. **Feature Extraction:** Convert text into numerical features using TF-IDF.
4. **Model Training:** Train a Logistic Regression classifier.
5. **Model Evaluation:** Evaluate performance using unseen benchmark test data.
6. **Prediction:** Analyze user-submitted news articles using the trained model.

## 📊 Model Performance

| Metric | Result |
|---|---:|
| Classification Algorithm | Logistic Regression |
| Feature Extraction | TF-IDF |
| Test Accuracy | 92.1% |

*Accuracy represents the reported benchmark test result. It does not guarantee that every real-world prediction is correct.*

## 📂 Project Structure

```text
NewsLens/
│
├── app.py
├── database.py
├── preprocess.py
├── train_model.py
├── test_system.py
├── requirements.txt
├── README.md
│
├── assets/
│   └── logo.png
│
├── data/
│   └── history.db
│
└── models/
    ├── news_model.pkl
    ├── tfidf_vectorizer.pkl
    └── model_metrics.json
```

*Note: The large training dataset is excluded from the GitHub repository.*

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/snehakumari1984/NewsLens.git
```

### 2. Navigate to the Project Directory

```bash
cd NewsLens
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Environment

Windows:

```bash
venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser at:

```text
http://localhost:8501
```

## 🚀 Deployment

NewsLens AI is deployed using **Streamlit Community Cloud**.

Access the live application:

https://newslens-hfysjkhpbctwzd3hwftrj.streamlit.app/

## 🔮 Future Improvements

- Integrate transformer-based NLP models.
- Improve model generalization across different news sources.
- Add multilingual news analysis.
- Integrate reliable external fact-checking sources.
- Improve explainability of machine learning predictions.
- Add more comprehensive model evaluation.

## ⚠️ Disclaimer

NewsLens AI is an experimental machine learning project developed for educational and research purposes.

Its predictions are based on learned textual patterns and should not be treated as definitive verification of factual accuracy. Users should consult reliable sources before accepting or sharing news.

## 👩‍💻 Project Information

**Project Name:** NewsLens AI  
**Domain:** Artificial Intelligence and Data Science  
**Focus Areas:** Natural Language Processing, Machine Learning, Fake News Detection

---

⭐ If you find this project interesting, consider starring the repository!
