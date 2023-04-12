# Credit risk classification among peer-to-peer loans

## Overview of the Analysis
The purpose of this analysis is to evaluate the efficacy of logistic regression modelling to detect risky loans from a sample of approximately 77,000 loans from a peer-to-peer lending service. Further, the analysis explores whether recalibrating the model using oversampling techniques improved the accuracy of the model.

The objective was to find out whether instances of 'healthy' and 'risky' loans could be correctly identified using machine learning.

The initial dataset contains 7 quantitative predictors:``loan_size``, ``interest_rate``, ``borrower_income``, ``debt_to_income``, ``num_of_accounts``, ``derogatory_marks`` and ``total_debt.``

The binary target variable ``loan_status`` comprised of two outcomes 0 - 'healthy loan' and 1 - 'risky loan.'

All predictor variables were saved to a 2D array variable 'X', and the target variable was saved to a data series 'y'.

The analysis employed sklearn's 'try_test_split' method to create subsets of X and y to use for training and testing purposes respectively. This was done because you can't evaluate the predictive performance of a model with the same data that you've used for training. The training subset is fitted to find the optimal weights and coefficients for the fresh testing data.

This creates 4 subsets of the data: X_train, X_test, y_train, y_test.

Training data was fitted to the logistic regression instance, then y-predictions were generated based on X-test data.

Because of a heavy skew towards 'healthy' loans in the dataset, the model was run a second time using the randomoversampling technique, which evens out the data by taking an equivalent number of samples from the minority class.

The results for both instances of the model are as follows.

## Results
* Machine Learning Model 1:
  * The overall accuracy of Model 1 is 99.24%
  * Predictive performance for 'healthy' loan detection was very strong in both precision and recall, with an f1-score of 99.61%.
  * Of all 18,746 'healthy' loan predictions, only 67 were actually 'risky', representing a precision rate of 99.64%
  * The model predicted 18,679 of the 18,759 'healthy' loans, a recall rate of 99.57%

  * Predictive performance for 'risky' loan detection was somewhat lower, with combined precision and recall (f1) score of 88.36%
  * Of all 638 'risky' predictions, 558 were correct, and 80 were False Positive, representing a precision rate of 89.28% 
  * Of the 625 'risky' loans, 558 were detected, representing a recall of 89.25%

* The discrepancy between the macro avg f1 score (93.98%) and the weighted-avg f1 score (99.25%) suggests that the distribution of labels is lopsided. For this reason, a second model was built with oversampled data.



* Machine Learning Model 2:
  * The overall accuracy score for Model 2 was 99.52%
  * Predictive performance for 'healthy' loan detection was slightly improved in model 2 to 99.75%
  * There were only 2 instances where the model predicted a 'healthy' loan when it was actually risky - a 99.99% precision rate.
  * Out of 18,759 instances of 'healthy' loans, the model detected 18,668 of them, a recall rate of 99.51% Very slightly down from Model 1.

  * The predictive performance of 'risky' loan detection was improved in model 2, with an f1 score of 93.05%
  * There were 91 instances where the model predicted a 'risky' loan when it was actually 'healthy'. This is an increase in false positives from Model 1, and represents a precision rate of 87.25%.
  * The model predicted 623 of the 625 risky loans, a recall rate of 99.68%

## Summary
* Both models appear to have  a high level of accuracy, with the second model slightly out performing the first overall.

* The second model, which uses random oversampling to balance the classes, had a higher f1 score of 93.05% when predicting 'risky' loans. Though the number of false positives increased slightly, representing a slight decrease in precision from 89.29% to 87.25%

* The higher False Positive rate when attempting to predict '1's highlights the difficulty in detecting 'risky' loans in such a lopsided dataset.

* Much of the improvement in Model 2's ability to detect '1's was in its recall. I.e. of all 625 'risky' loans, Model 2 predicted 623.

* Given the risks associated with a False Positive in this scenario are relatively small (i.e. incorrectly labelling a loan as 'risky' is not as detrimental as say, a diabetes diagnosis), it is recommended that the improvements to Model 2's recall make it the superior model.