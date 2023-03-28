# huhu23
Model to participate un HUHU-2023 Iberlef


## Results

| encoder                                           |   vector selection   | Weighted sample | last layer model | F-1    | ACC    | MSE    |
|---------------------------------------------------|:--------------------:|-----------------|------------------|--------|--------|--------|
| idf-tf                                            |           -          |      No         | Random Forest    | 0.7122 | 0.7852 | -      |
| idf-if                                            |           -          |      No         | Random Forest    | -      | -      | 0.3894 |
| hackathon-pln-es/paraphrase-spanish-distilroberta | MaxPool with AttMask |      No         | Random Forest    |        |        | 0.3527 |
| hackathon-pln-es/paraphrase-spanish-distilroberta | MaxPool with AttMask |      No         | Random Forest    | 0.7619 | 0.8074 |        |