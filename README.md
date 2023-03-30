# huhu23
Model to participate un HUHU-2023 Iberlef


## Results

| encoder                                           |   vector selection   |  Data balancing  | last layer model | F-1    | ACC    | MSE    |
|---------------------------------------------------|:--------------------:|:----------------:|------------------|--------|--------|--------|
| idf-tf                                            |           -          |      No          | Random Forest    | 0.7122 | 0.7852 | -      |
| idf-tf                                            |           -          |      SMOTE       | Random Forest    | 0.7300 | 0.7926 | -      |
| idf-if                                            |           -          |      No          | Random Forest    | -      | -      | 0.3894 |
| hackathon-pln-es/paraphrase-spanish-distilroberta | MaxPool with AttMask |      No          | Random Forest    | 0.7619 | 0.8074 |        |
| hackathon-pln-es/paraphrase-spanish-distilroberta | MaxPool with AttMask |      SMOTE       | Random Forest    | 0.7868 | 0.8222 |        |
| hackathon-pln-es/paraphrase-spanish-distilroberta | MaxPool with AttMask |      No          | Random Forest    |        |        | **0.3527** |
| pysentimiento/robertuito-sentiment-analysis       | MaxPool with AttMask |      No          | Random Forest    | 0.7436 | 0.7926 |        |
| pysentimiento/robertuito-sentiment-analysis       | MaxPool with AttMask |      SMOTE       | Random Forest    | 0.7952 | 0.8222 |        |
| pysentimiento/robertuito-sentiment-analysis       | MaxPool with AttMask |      No          | Random Forest    |        |        | 0.4272 |
| edumunozsala/beto_sentiment_analysis_es           | MaxPool with AttMask |      No          | Random Forest    | 0.6859 | 0.7778 |        |
| edumunozsala/beto_sentiment_analysis_es           | MaxPool with AttMask |      SMOTE       | Random Forest    | 0.7115 | 0.7630 |        |
| edumunozsala/beto_sentiment_analysis_es           | MaxPool with AttMask |      No          | Random Forest    |        |        | 0.4038 |
| cardiffnlp/twitter-xlm-roberta-base               | MaxPool with AttMask |      No          | Random Forest    | 0.7465 | 0.8000 |        |
| cardiffnlp/twitter-xlm-roberta-base               | MaxPool with AttMask |      SMOTE       | Random Forest    | 0.7652 | 0.8000 |        |
| cardiffnlp/twitter-xlm-roberta-base               | MaxPool with AttMask |      No          | Random Forest    |        |        | 0.4340 |
| Manauu17/enhanced_roberta_sentiments_es           | MaxPool with AttMask |      No          | Random Forest    | 0.7278 | 0.7852 |        |
| Manauu17/enhanced_roberta_sentiments_es           | MaxPool with AttMask |      SMOTE       | Random Forest    | **0.7999** | **0.8296** |        |
| Manauu17/enhanced_roberta_sentiments_es           | MaxPool with AttMask |      No          | Random Forest    |        |        | 0.4296 |
| daveni/twitter-xlm-roberta-emotion-es             | MaxPool with AttMask |      No          | Random Forest    | 0.7547 | 0.8000 |        |
| daveni/twitter-xlm-roberta-emotion-es             | MaxPool with AttMask |      SMOTE       | Random Forest    | 0.7826 | 0.8148 |        |
| daveni/twitter-xlm-roberta-emotion-es             | MaxPool with AttMask |      No          | Random Forest    |        |        | 0.4296 |
