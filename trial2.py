#testing the logistic regression sci-kitlearn
#trial2.py
#http://nbviewer.ipython.org/github/justmarkham/gadsdc1/blob/master/logistic_assignment/kevin_logistic_sklearn.ipynb

import numpy as np
#import pandas as pd
import matplotlib
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score


def main():
    #print "test"
    X = [[1,0], [2, 0], [2, 1], [4, 7], [-1, 3], [-2, 0]]
    #X = [[1,0,1]]
    #(X,y) = buildTrainingData(inputFile....)
    y = np.array([1, 1, 1, 0, 0, 0])

    model = LogisticRegression()
    model = model.fit(X, y)

    # check the accuracy on the training set
    #print model.score(X, y)
    #print y.mean()
    print model.predict_proba([0,3])
    

#def ReadFileInAsMatrix():


if __name__ == "__main__":
    main()
