Underfitting and overfitting can be avoided in this problem by:

- Using cross-validation, such as K-folds, in our train/test split;
- Tune hyperparameters using only the training set (leaving the test set unseen);
- Resampling using SMOTE method (e.g., "intallment_status" had 60% missing data);
- More historical data. Ideally more two years of information would be nice to begin testing for annual seasonality 
and perform time series analysis. This also can be used to apply rolling window validation.
- Combine different algorithms into one;
- Regularization (see ElasticNet algorithm).


The model would be better automated if the data it needs were available in a DataLake, such as Amazon S3, Azure, Oracle.

I remind that in production environment, we would probably rewrite the code in PySpark, since it runs
smoothier in BigData (parallelization).

Model's production environment should be isolated from development environment.

An ETL query would inform the orchestrator, such as Docker or Azure Data Factory the procedure and frequency of
data flow. Small but frequent updates are preferable over big and casual ones. CI/CD (continuous integration/continuous delivery)
framework.
