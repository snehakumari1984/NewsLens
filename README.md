# 🔎 NewsLens AI – AI-Powered News Credibility Analysis System

\<p align="center">   \<strong>Detecting Potentially Misleading News Using Natural Language Processing and Machine Learning\</strong> \</p>

\<p align="center">   🚀 \<a href="[https://newslens-ne6cdpjqz6t62nfbpxbdoc.streamlit.app/](https://newslens-ne6cdpjqz6t62nfbpxbdoc.streamlit.app/)">\<strong>Live Demo\</strong>\</a> |   💻 \<a href="[https://github.com/snehakumari1984/NewsLens](https://github.com/snehakumari1984/NewsLens)">\<strong>GitHub Repository\</strong>\</a> \</p>

---

## 📌 Project Overview

NewsLens AI is an AI-powered news credibility analysis system developed using Natural Language Processing (NLP) and supervised Machine Learning.

The application analyzes the linguistic and lexical patterns of news articles to predict whether the submitted content is likely to be real or potentially misleading.

It uses TF-IDF (Term Frequency–Inverse Document Frequency) for text feature extraction and a Logistic Regression classifier for prediction.

NewsLens provides an interactive Streamlit dashboard where users can submit news headlines and article content, explore predictions, and view model performance statistics.

**The main objective is to demonstrate how machine learning can support the analysis of digital news content and help users become more aware of potentially misleading information.**

## 🚀 Live Demo

Try the deployed application:

👉 [https://newslens-ne6cdpjqz6t62nfbpxbdoc.streamlit.app/](https://newslens-ne6cdpjqz6t62nfbpxbdoc.streamlit.app/)

## ✨ Key Features

- **News Credibility Analysis:** Analyzes news headlines and article content.
- **Machine Learning Prediction:** Uses Logistic Regression for text classification.
- **NLP-Based Feature Extraction:** Applies TF-IDF vectorization to convert text into numerical features.
- **Interactive Dashboard:** Provides an intuitive and modern user interface.
- **Prediction History:** Stores local analysis records using SQLite.
- **Model Performance:** Displays model accuracy and evaluation information.
- **Dataset Visualization:** Presents training dataset distribution and model comparison information.
- **Sample News Articles:** Includes sample articles to demonstrate the analysis system.

## 🧠 Machine Learning Approach

NewsLens AI follows a supervised machine learning workflow.

### 1. Text Preprocessing

News text is prepared for feature extraction.

### 2. Feature Extraction

TF-IDF vectorization converts textual information into numerical features.

### 3. Model Training

A Logistic Regression classifier is trained to identify patterns associated with the available news categories.

### 4. Prediction

The trained model analyzes submitted news content and generates a classification.

### 5. Model Evaluation

The model is evaluated using benchmark test data.

### Model Details

| Component                | Technology          |
| ------------------------ | ------------------- |
| Classification Algorithm | Logistic Regression |
| Feature Extraction       | TF-IDF              |
| Machine Learning         | Scikit-learn        |
| Test Accuracy            | 92.1%               |

*The reported accuracy represents the project's benchmark test result. It does not guarantee the correctness of individual predictions.*

## 🛠️ Technologies Used

| Technology          | Application                 |
| ------------------- | --------------------------- |
| Python              | Core programming            |
| Streamlit           | Web application development |
| Scikit-learn        | Machine Learning            |
| Pandas              | Data processing             |
| NumPy               | Numerical computing         |
| TF-IDF              | Text feature extraction     |
| Logistic Regression | Text classification         |
| SQLite              | Prediction history storage  |
| Plotly              | Data visualization          |
| Joblib              | Model serialization         |

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
├── .gitignore
│
├── assets/
│   └── logo.png
│
├── data/
│   └── history.db
│
└── models/
    ├── model_metrics.json
    ├── news_model.pkl
    └── tfidf_vectorizer.pkl
```

## ⚙️ Installation and Setup

Follow these instructions to run the project on your local machine.

### Step 1: Clone the Repository

```bash
git clone https://github.com/snehakumari1984/NewsLens.git
```

### Step 2: Navigate to the Project Directory

```bash
cd NewsLens
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate the Virtual Environment

For Windows:

```bash
venv\Scripts\activate
```

### Step 5: Install Required Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Run the Application

```bash
streamlit run app.py
```

The application will be available at:

`http://localhost:8501`

## 📊 Model Performance

The deployed dashboard currently reports:

| Performance Indicator    |              Result |
| ------------------------ | ------------------: |
| Test Accuracy            |               92.1% |
| Classification Algorithm | Logistic Regression |
| Feature Extraction       |              TF-IDF |

## 🔮 Future Enhancements

- Integration with trusted news sources for additional verification.
- Multilingual news credibility analysis.
- Experimentation with advanced NLP architectures such as BERT.
- Explainable AI for understanding prediction results.
- Improved detection of misleading and manipulated content.
- Real-time news analysis capabilities.

## ⚠️ Disclaimer

NewsLens AI is an educational machine learning project.

The system identifies patterns in text based on its training data. Its predictions should not be considered definitive proof of whether a news article is true or false.

Users should independently verify important information using reliable news sources.

## 👩‍💻 Developer

**Sneha Kumari**

B.Tech – Artificial Intelligence and Data Science

GitHub: [@snehakumari1984](https://github.com/snehakumari1984)

---

\<p align="center">
&#x20; ⭐ If you find this project interesting, consider giving this repository a star!
\</p>
