1. Overview
	Given a dataset of records on drug reviews, develop algorithms to predict the rating level of a drug
	given by a review.
2. Data Description
	Here used the training data to explain thedata format, as follows.
	training.csv
	- Each row represents the information of a review on a drug.
	- Each row contains following attributes
		o recordId: record id of the review
		o drugName: name of the drug
		o condition: condition to take the drug
		o reviewComment: review comment
		o date: review created date
		o usefulCount: the number of users who find the review useful
		o sideEffects: level of side effects of the drug
	- The last cell of each row is the `rating` level of the review
		o The rating level of each review is the target to predict
		o The rating levels are integers in 1, 2, 3, 4, 5 classes
		o Higher rating indicates more positive opinion

	The data_source file contains three dataset: training.csv, validation.csv, testing.csv
3. Approaches
	This dataset can use different approaches to archeive a better prediction result.
   	Here the evaluation metrics using are Micro-F1 and Macro F1 score.
	- TFIDF_SVD_NN
	- To be continued
