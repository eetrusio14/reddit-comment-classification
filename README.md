# Classifying Reddit Comments Across 40 Subreddits

A comparative machine learning study evaluating four classification algorithms on 500,000 Reddit comments, built using PySpark on AWS EMR.

## Overview

Text classification on social media is a challenging multi-class problem due to informal, noisy, high-dimensional user-generated content. This project predicts the source subreddit of a given comment across 40 diverse communities (e.g. r/gameofthrones, r/nba, r/relationship_advice), against a random-guess baseline of just 2.5%.

Four models were trained and compared:
- Logistic Regression
- Naive Bayes
- Decision Tree
- Multilayer Perceptron (MLP)

## Dataset & Feature Engineering

- **Size:** 500,000 Reddit comments, stored on AWS S3
- **Processing:** PySpark on an AWS EMR cluster
- **Pipeline:** null/non-alphanumeric cleaning, class balancing (capped at 12,500 rows per subreddit), tokenization, stop word removal, CountVectorizer, and TF-IDF weighting
- **Final feature space:** 15,002 dimensions

## Methodology

| Model | Configuration |
|---|---|
| Logistic Regression | 30 iterations, regParam = 0.01 |
| Naive Bayes | Multinomial, Laplace smoothing |
| Decision Tree | Max depth = 8 |
| MLP | Two hidden layers (128, 64 nodes), full 15,002-feature input |

Models were evaluated on Accuracy and Macro F1-score to weight all 40 classes equally.

## Results

| Model | Accuracy | F1-Score |
|---|---|---|
| **Logistic Regression** | **0.4101** | **0.4132** |
| Naive Bayes | 0.3636 | 0.3524 |
| MLP Neural Network | 0.3204 | 0.3087 |
| Decision Tree | 0.0598 | 0.0577 |

Logistic Regression performed best, achieving an F1-score roughly 16 times the random baseline. This suggests subreddit classification in TF-IDF space is approximately linear. Interpreting the model's weights shows it picks up on community-specific vocabulary — e.g. r/gameofthrones is predicted by character names, while r/AmItheAsshole is defined by acronyms like "YTA" and "NTA". The MLP and Decision Tree underperformed, likely due to the high dimensionality and sparsity of the TF-IDF feature space.

## Applications & Ethical Considerations

This kind of classifier has practical use in automated content moderation and community recommendation systems. It also carries risks, including potential for targeted harassment or biased moderation outcomes. The interpretability of the winning model supports transparency in how such decisions are made, which matters for maintaining user trust in automated systems.

## Repository Contents

- `Final.ipynb` — full PySpark pipeline: data loading and cleaning, feature engineering, model training/evaluation, and visualizations
- `Reddit_Classification_Presentation_Final_Project.pptx` — final presentation slides

## My Contributions

This was a group project completed with two teammates. My individual contributions:
- Built and executed the full Jupyter notebook pipeline on AWS EMR
- Created the project pitch presentation
- Developed all visualizations (Seaborn/Matplotlib) covering model performance and feature interpretability

All three team members collaborated on problem formulation, the final presentation, and overall project design.

## Tech Stack

PySpark · AWS EMR · AWS S3 · scikit-learn · Pandas · Seaborn · Matplotlib

## Future Work

Exploring transformer-based embeddings and n-gram features to improve on the current 41.3% F1-score ceiling.
