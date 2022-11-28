#!/usr/bin/env python
# coding: utf-8

import numpy as np
from numpy.random import randn
from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification, make_circles
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn import tree
from sklearn.metrics import accuracy_score, classification_report, average_precision_score, auc, roc_auc_score, roc_curve, confusion_matrix
from sklearn.decomposition import PCA

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
import graphviz

# use StandardScaler

def scale(XTrain, XTest):
    scaler = StandardScaler()
    scaler.fit(XTrain)
    XTrainScaled = scaler.transform(XTrain)
    XTestScaled = scaler.transform(XTest)
    
    return XTrainScaled, XTestScaled


# In[76]:


def evaluationReport(CLTrain, CL_pred_Train, CLTest, CL_pred_Test, isBinary = True):

    ## raw accuracy score    
    accTrain = accuracy_score(CLTrain, CL_pred_Train)
    accTest = accuracy_score(CLTest, CL_pred_Test)

    ## cross-val score

    print('logit classification accuracy on training set: {:.2f}'.format(accTrain))
    print('logit classification accuracy on test set: {:.2f}'.format(accTest))
    
    print('full classification report (on test set:)')
    print(classification_report(CLTest, CL_pred_Test))
    
    if isBinary:
	    print('average precision score: %0.2f' % average_precision_score(CLTest, CL_pred_Test))
	    
    print("confusion matrix: \n",confusion_matrix(CLTest, CL_pred_Test))
    if isBinary:
       tn, fp, fn, tp = confusion_matrix(CLTest, CL_pred_Test).ravel()
       print("tn: %0.2f, fp: %0.3f, fn: %0.2f, tp: %0.2f" % (tn, fp, fn, tp))

	    
def downsample(X,CL):
    
    ## we want to achieve roughly 50% contribution for each class
    currentRatio = Counter(CL)[0] / Counter(CL)[1]
    print("current class labels ratio: %0.2f" % currentRatio)

    if currentRatio < 1:
        majority = 1
        threshold = 1- currentRatio
    else:
        majority = 0
        threshold = 1 - 1/ currentRatio
    
    n = 0
    X_reb = np.arange(0).reshape(0, X.shape[1])
    CL_reb = np.arange(0)
    for i in range(len(CL)):
        if CL[i] == majority and randn() <= threshold:
            ## removing record
            n +=1
        else:
            ## copying record
            CL_reb = np.append(CL_reb, CL[i])
            X_reb = np.append(X_reb, X[i:1+i],  axis=0)
#             print("X[%d]" % i) 
#             print(X[i])
#             print(X_reb)

    print("X_reb length: ", len(X_reb))
    print("initially: ", Counter(CL))
    print("majority class: %d" % majority)
    print("threshold: %0.2f" % threshold)
    print("%d majority class records removed "% n)
    print("%d majority class records remaining" %(len(CL_reb)))
    print("new class labels ratio: %0.2f" % (Counter(CL_reb)[0] / Counter(CL_reb)[1]))
    print("counts: ",Counter(CL_reb))
    
    return X_reb, CL_reb


# In[77]:


def logit(XTrain, CLTrain, XTest, CLTest, penalty='l2'):

    # generate model using training set, and evaluate using test set

    clf = LogisticRegression(penalty=penalty, C=1, solver='lbfgs',multi_class='ovr')
    clf.fit(XTrain, CLTrain)

    # predictions on training set
    CL_pred_Train = clf.predict(XTrain)

    # predictions on test set
    CL_pred_Test = clf.predict(XTest)

    return clf, CL_pred_Train, CL_pred_Test


# In[78]:


# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
def LinearSVM(XTrain, CLTrain, XTest, CLTest):
    
    svm = LinearSVC(C=1, loss= "hinge")
    clf = svm.fit(XTrain, CLTrain)

    # predictions on training set
    CL_pred_Train = clf.predict(XTrain)

    # predictions on test set
    CL_pred_Test = clf.predict(XTest)

    return clf, CL_pred_Train, CL_pred_Test
    


# In[79]:


def SVM(XTrain, CLTrain, XTest, CLTest, kernel):
    
    svm = SVC(kernel = kernel, degree = 3, C=5, coef0=1)
    clf = svm.fit(XTrain, CLTrain)

    # predictions on training set
    CL_pred_Train = clf.predict(XTrain)

    # predictions on test set
    CL_pred_Test = clf.predict(XTest)

    return clf, CL_pred_Train , CL_pred_Test


# In[80]:


def plotTrainTest(XTrain, CLTrain, XTest, CLTest):
    fig = plt.figure(figsize=(20,6))
    fig.subplots_adjust(hspace=1, wspace=0.4)

    ax = fig.add_subplot(1, 2, 1)
    sns.scatterplot(x=XTrain[:,0],y=XTrain[:,1], hue=CLTrain, ax=ax)  # plot training set
    ax.set_title("training set")

    ax = fig.add_subplot(1,2, 2)
    sns.scatterplot(x=XTest[:,0],y=XTest[:,1], hue=CLTest, ax=ax)
    ax.set_title("test set")
    plt.show()


# In[81]:


def plotLinearFitTrainTest(clf, XTrain, CLTrain, XTest, CLTest):
    
    x_Train_min, x_Train_max = XTrain[:, 0].min() - .5, XTrain[:, 0].max() + .5
    x_Test_min, x_Test_max = XTest[:, 0].min() - .5, XTest[:, 0].max() + .5

    x_min = min(x_Train_min, x_Test_min)
    x_max = max(x_Train_max, x_Test_max)
            
        
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(x_min, x_max)
    yy = a * xx - (clf.intercept_[0]) / w[1]

    fig = plt.figure(figsize=(20,6))
    fig.subplots_adjust(hspace=1, wspace=0.4)

    ax = fig.add_subplot(1, 2, 1)
    sns.scatterplot(x=XTrain[:,0],y=XTrain[:,1], hue=CLTrain)  # plot training set
    ax.set_title("training set with separation line")
    ax2 = ax.twinx()
    sns.regplot(x=xx,y=yy, ax=ax2)

    ax = fig.add_subplot(1,2, 2)
    sns.scatterplot(x=XTest[:,0],y=XTest[:,1], hue=CLTest)
    ax.set_title("test set with separation line")
    ax2 = ax.twinx()
    sns.regplot(x=xx,y=yy, ax= ax2)
    plt.show()
    


# In[82]:


def plotContourFitTrainTest(clf, XTrain, CLTrain, XTest, CLTest):

    h = .02  # step size in the mesh

    fig = plt.figure(figsize=(20,6))
    fig.subplots_adjust(hspace=1, wspace=0.4)
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])


    ## plot training set
    x_min, x_max = XTrain[:, 0].min() - .5, XTrain[:, 0].max() + .5
    y_min, y_max = XTrain[:, 1].min() - .5, XTrain[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    ax = fig.add_subplot(1, 2, 1)
    sns.scatterplot(x=XTrain[:,0],y=XTrain[:,1], hue=CLTrain, ax=ax)  # plot training set
    ax.set_title("training set wih contour line")
    
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
    
    
    ## plot test set
    x_min, x_max = XTest[:, 0].min() - .5, XTest[:, 0].max() + .5
    y_min, y_max = XTest[:, 1].min() - .5, XTest[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    ax = fig.add_subplot(1, 2, 2)
    sns.scatterplot(x=XTest[:,0],y=XTest[:,1], hue=CLTest, ax=ax)  # plot training set
    ax.set_title("test set wih contour line")
    
    # Plot the decision boundary.
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)


def plotContourFitTrainTestAlternate(clf, XTrain, CLTrain, XTest, CLTest, sharp=False):

    # probability decision surface for logistic regression on a binary classification dataset
    from numpy import where
    from numpy import meshgrid
    from numpy import arange
    from numpy import hstack
    from matplotlib import pyplot
    
    h = .1  # step size in the mesh

    ## plot training set
    x_min, x_max = XTrain[:, 0].min() - .5, XTrain[:, 0].max() + .5
    y_min, y_max = XTrain[:, 1].min() - .5, XTrain[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # flatten each grid to a vector
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))

    # horizontal stack vectors to create x1,x2 input for the model
    grid = hstack((r1,r2))

    # make predictions for the grid
    if sharp:
        yhat = clf.predict(grid)
    else:
        yhat = clf.predict_proba(grid)[:, 1]
            
    # reshape the predictions back into a grid
    zz = yhat.reshape(xx.shape)

    # plot the grid of x, y and z values as a surface
    c = pyplot.contourf(xx, yy, zz, cmap='Paired')

    # add a legend, called a color bar
    if sharp:
        pyplot.colorbar(c)

    # create scatter plot for samples from each class
    for class_value in range(2):
        # get row indexes for samples with this class
        row_ix = where(CLTrain == class_value)
        # create scatter of these samples
        pyplot.scatter(XTrain[row_ix, 0], XTrain[row_ix, 1], cmap='Paired')

    # show the plot
    pyplot.show()

  
    
def plotROC(clf, XTest, CLTest, CL_pred_Test=None):
    
    if hasattr(clf, "decision_function"):
        print("using decision_function")
        probs = clf.decision_function(XTest)
        preds = probs
    else:
        print("using predict_proba")
        probs = clf.predict_proba(XTest)
        preds = probs[:,1]

    fpr, tpr, threshold = roc_curve(CLTest, preds)
    roc_auc = auc(fpr, tpr)

    if CL_pred_Test is not None:
        print("\n\n====== ROC ======")
        print("roc_auc_score = %0.2f" % roc_auc_score(CLTest, CL_pred_Test))
        print("auc = %0.2f" % roc_auc)

    fig = plt.figure()
    plt.title('ROC curve')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

# In[110]:


# https://towardsdatascience.com/synthetic-data-generation-a-must-have-skill-for-new-data-scientists-915896c0c1ae
# see https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_circles.html#sklearn.datasets.make_circles
# and https://scikit-learn.org/stable/datasets/index.html#sample-generators

def makeDataset(kind='classification', sameScale=True, balanced = True, unbalance = 0.5, n_classes = 2, n_features=2, n_samples=1000, n_clusters_per_class=2, n_informative=2):

    if kind == 'classification':
        X, CL = make_classification(n_samples=n_samples, n_classes=n_classes, n_features=n_features, 
                                   n_redundant=0, n_informative=n_informative, random_state=5, n_clusters_per_class=n_clusters_per_class,
                                   class_sep = 1,
                                   flip_y = 0.1)
    elif kind == 'circle':
        X, CL = make_circles(n_samples=1000, noise=0.1, factor=.5, random_state=1)

    if not sameScale:
         X[:,0] = X[:,0] * 100
            
    if not balanced:
        n =0
        for i in range(len(CL)):
            if CL[i] == 0:
                if randn() <= unbalance:
                    n +=1
                    CL[i] = 1
        print(n," CL values flipped")
        print("class labels ratio: %0.2f" % (Counter(CL)[0] / Counter(CL)[1]))
    return X, CL

