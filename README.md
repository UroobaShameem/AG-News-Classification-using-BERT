# AG News Classification Project

## Table of Contents
1. [Introduction](#introduction)
2. [Project Objectives](#project-objectives)
3. [Dataset](#dataset)
4. [Model Used](#model-used)
   - [BERT Fine-Tuning](#bert-fine-tuning)
   - [Model Performance](#model-performance)
5. [Tools and Libraries Used](#tools-and-libraries-used)
6. [How to Run the Code](#how-to-run-the-code)

---

## Introduction
News classification is a critical natural language processing (NLP) task used to categorize articles into specific topics, helping users quickly access content relevant to their interests. This project, developed as part of a course at the University of Karachi, applies a state-of-the-art NLP model (BERT) to classify news articles into four predefined categories:
- **World**
- **Sports**
- **Business**
- **Science/Technology**

Using the BERT model, which is based on Transformer architecture, we leverage advanced contextual embeddings to analyze and classify articles efficiently. This project was divided into several steps, including data preprocessing, model fine-tuning, evaluation, and performance analysis on unseen data. The resulting BERT model provides high accuracy, demonstrating its suitability for text classification tasks.

## Project Objectives
1. **Develop a BERT-based classification model** using Hugging Face's transformers library.
2. **Evaluate model performance** in classifying news articles accurately across four categories.

## Dataset
The dataset consists of:
- **Training set**: 8,000 articles, balanced across four categories (2,000 per category).
- **Test set**: 100 articles for model evaluation.

Each article entry includes:
- `Title`: Headline of the news article.
- `Description`: Brief description of the content.
- `Class Index`: Label indicating the category.

## Model Used

### BERT Fine-Tuning
- **Library**: Hugging Face's `transformers`
- **Preprocessing**: Data cleaning, tokenization, and label adjustment.
- **Training Parameters**:
  - Learning Rate: `2e-5`
  - Batch Size: `16`
  - Epochs: `3`
- **Evaluation Metrics**: Precision, Recall, F1-score, and Confusion Matrix.

### Model Performance
| Metric     | BERT Model |
|------------|------------|
| Accuracy   | 97%        |
| Precision  | 96%        |
| Recall     | 96%        |
| F1-Score   | 96%        |

### Key Insights
- The **BERT model** performed exceptionally well in classification metrics and is optimized for sequence classification.

## Tools and Libraries Used
- **Transformers** (Hugging Face): BERT model handling.
- **PyTorch**: Model training backend.
- **NLTK**: Preprocessing tasks (e.g., stopword removal, punctuation cleaning).
- **Sklearn**: Evaluation metrics such as precision, recall, and F1-score.
- **Matplotlib/Seaborn**: Visualization of class distributions and confusion matrix.

## How to Run the Code

### Prerequisites
Ensure you have Python installed (recommended version 3.6+). You will also need to install the following packages:
- `transformers`
- `torch`
- `nltk`
- `sklearn`
- `matplotlib`

You can install these dependencies with the following command:

```bash
pip install transformers torch nltk sklearn matplotlib
```

### Running the Code
1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd AG_News_Classification
   ```

2. **Prepare the Dataset**:
   - Place the `train.csv` and `test.csv` files in the `data` folder within the project directory.
   - Ensure each file follows the format with columns: `Title`, `Description`, and `Class Index`.

3. **Run the Jupyter Notebook**:
   - Open `AG_News_Classification.ipynb` in Jupyter Notebook or JupyterLab.
   - Execute the cells in sequence to preprocess the data, fine-tune the BERT model, and evaluate the results.

4. **View Results**:
   - Evaluation metrics (precision, recall, F1-score) and confusion matrix will be displayed within the notebook upon completion of the training and evaluation cells.

