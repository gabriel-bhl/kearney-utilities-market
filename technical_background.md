### 1. How do you avoid overfitting and underfitting in this problem?

In this problem, the biggest concern regarding under/overfitting was with the "installment_status" feature, which had
~60% missing values.

In general, underfitting and overfitting can be avoided by:

- Using cross-validation, such as K-folds, in our train/test split, i.e., we change them every "round", leaving an unseen
validation set, K times;
- Tune hyperparameters using only the training set (leaving the test set unseen);
- Resampling using SMOTE method;
- More historical data. Ideally more two years of information would be nice to begin testing for annual seasonality 
and perform time series analysis. This also can be used to apply rolling window validation.
- Combine different algorithms into one;
- Regularization (see ElasticNet algorithm).




### 2. Explain shortly how you would transfer your model to the client’s IT Department. (Hint: DevOps)
The model itself would be in a shared repository (either GitHub, GitLab, Azure DevOps) with one administrator responsible 
for keeping the `main` branch uptaded (via pull requests from developers).

(I remind that in production environment, we would probably rewrite the code in PySpark, since it runs
smoothier in BigData (parallelization).)

A query should be written in order to perform the ETL pipeline from the data source (Data Lakes such as Amazon S3,
Azure, Oracle) to the production environment (Azure Data Factory, a Docker/Kubernetes etc.).

Model's production environment should be isolated from development environment.

Small but frequent updates are preferable over big and casual ones. CI/CD (continuous integration/continuous delivery)
framework.
