from math import exp
from scipy.stats import pearsonr
from sklearn.gaussian_process import GaussianProcess
from sklearn.cross_validation import train_test_split
from scipy.stats import kruskal
import matplotlib.pyplot as plt
import random
import numpy as np

class causation(object):
    def __init__(self,X,Y):
        self.X = np.asarray([X]).T
        self.Y = np.asarray([Y]).T
        print self.X, self.X

    def correlation_score(self,criteria='pearsonr'):
        return {
            'pearsonr': self.pearsonr_correlation(self.X,self.Y),
        }.get(criteria, AttributeError('Unknown criteria ' + criteria))


    def pearsonr_correlation(self,X,Y):
        correlation_coef, p_val = pearsonr(X,Y)
        return abs( correlation_coef[0] )

    def ANM_predict_causality(self,train_size=0.7,independence_criterion='kruskal'):
        '''
            Prediction of causality based on the bivariate additive noise model

            Parameters
            ----------
            independence_criterion : kruskal for Kruskal-Wallis H-test

            Returns
            -------
            Causal-direction: 1 if X causes Y, or -1 if Y causes X
        '''
        Xtrain, Xtest , Ytrain, Ytest = train_test_split(self.X, self.Y, train_size = train_size)
        _gp = GaussianProcess()

        #Forward case
        _gp.fit(Xtrain,Ytrain)
        errors_forward = _gp.predict(Xtest) - Ytest
        forward_indep_score = 0.5

        #Backward case
        _gp.fit(Ytrain,Xtrain)
        errors_backward = _gp.predict(Ytest) - Xtest

        #Independence score

        forward_indep_pval = {
            'kruskal': kruskal(errors_forward,Xtest)[1]
        }[independence_criterion]

        backward_indep_pval = {
            'kruskal': kruskal(errors_backward,Ytest)[1]
        }[independence_criterion]

        #print 'Scores:', forward_indep_pval, backward_indep_pval

        if forward_indep_pval < backward_indep_pval:
            self.causal_direction = 1
            self.pvalscore = forward_indep_pval
        else:
            self.causal_direction = -1
            self.pvalscore = backward_indep_pval

        return {'causal_direction':self.causal_direction,'pvalscore':self.pvalscore}


    def plot_data(self):
        plt.scatter(self.X,self.Y)
        plt.show()


def QA_single(steps=50,dataset_fn='pair0001.txt'):
    fin = open(dataset_fn)
    A, B = [], []
    for l in fin:
        a, b = l.strip().split()
        if float(a) in A or float(b) in B: continue
        A.append(float(a))
        B.append(float(b))

    plt.scatter(A,B)
    plt.show()

    CO = causation(A,B)
    D = {1:[],-1:[]}
    for i in xrange(steps):
        d = CO.ANM_predict_causality()
        D[d['causal_direction']].append(d['pvalscore'])

    plt.hist([D[1],D[-1]],bins=30,label=['(X,Y)','(Y,X)'],alpha=0.5)
    plt.legend()
    plt.show()

    if len(D[1])> len(D[-1]):
        return 1
    else:
        return -1

def main():
    A, B = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [1, 4, 5, 3, 2, 9, 0, 8, 6, 7, 11, 13, 12]
    CO = causation(A,B)
    print 'A:', A
    print 'B:', B
    print 'correlation(A,B):', CO.correlation_score(criteria='pearsonr')
    print 'cause (A,B)?:', CO.ANM_predict_causality()
    CO.plot_data()

if __name__ == '__main__':
    main()
    if QA_single() == 1:
        print '\n\n X --> Y'
    else:
        print '\n\n Y --> X'
