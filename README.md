# Yummly
Machine Learning Project

Exploring Recipe Data

- Naive Bayes 
- SVM
- Decision Trees


###lr.py
Word lemmatization version 1 - better for logistic regression
cleaned up words and tfidf vectorized ingredients from the corpus
benchmark 50-50 split:
NB .715
LR .777
SVC .778

###nb2.py
Word lemmatization version 2 - better for NB
simple lowercase lemmatization
benchmark 50-50 split:
NB .739
LR .774
SVC .774

###nb.py
no lemmatization.
Includes code to do kfold testing.
Do not run kfold with gridsearch 
benchmark 50-50 split:
NB .731
LR .729
SVC .772

