from math import exp
from scipy.stats import pearsonr
from sklearn.gaussian_process import GaussianProcess
from sklearn.cross_validation import train_test_split
from scipy.stats import kruskal
import matplotlib.pyplot as plt
import random
import numpy as np
import os
from sklearn.kernel_ridge import KernelRidge
from scipy import stats

import warnings
warnings.filterwarnings('ignore')

class causation(object):
    def __init__(self,X,Y):
        self.X = np.asarray([X]).T
        self.Y = np.asarray([Y]).T
        #print self.X, self.X

    def correlation_score(self,criteria='pearsonr'):
        return {
            'pearsonr': self.pearsonr_correlation(self.X,self.Y),
        }.get(criteria, AttributeError('Unknown criteria ' + criteria))


    def pearsonr_correlation(self,X,Y):
        correlation_coef, p_val = pearsonr(X,Y)
        return abs( correlation_coef[0] )

    def ANM_predict_causality(self,train_size=0.4,independence_criterion='kruskal'):
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
        _gp = KernelRidge(kernel='sigmoid')#GaussianProcess(theta0=1000)

        #Forward case
        _gp.fit(Xtrain,Ytrain)
        errors_forward = _gp.predict(Xtest) - Ytest

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


    def forceAspect(self,ax,aspect=1):
        '''
            Just to force aspect ratio for figures
        '''
        im = ax.get_images()
        extent =  im[0].get_extent()
        ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)



    def plot_data_density(self,bw_method='scott'):
        '''
            Plot of data points and estimated joint density

            Parameters
            ----------
            bw_method : KDE estimator bandwidth. 'scott', 'silverman' or
                        scipy.stats.gaussian_kde instance

            Returns
            -------
            None
        '''
        m1, m2 = (self.X.T), (self.Y.T)
        eps1 = 0.5*np.std(m1)
        eps2 = 0.5*np.std(m2)
        xmin = m1.min() - eps1
        xmax = m1.max() + eps1
        ymin = m2.min() - eps2
        ymax = m2.max() + eps2
        X, Y = np.mgrid[xmin:xmax:50j, ymin:ymax:50j]
        positions = np.vstack([X.ravel(), Y.ravel()])
        values = np.vstack([m1, m2])
        kernel = stats.gaussian_kde(values,bw_method=bw_method)
        Z = np.reshape(kernel(positions).T, X.shape)
        fig = plt.figure(figsize=(11, 5))
        ax = fig.add_subplot(111)
        ax.imshow(np.rot90(Z), cmap=plt.cm.RdYlBu,
                   extent=[xmin, xmax, ymin, ymax])
        ax.plot(m1, m2, 'k.', markersize=5,linewidth='0.1',color='b',alpha=0.4)
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        self.forceAspect(ax,aspect=1)
        plt.show()

        return None

def AMM_bagging_causality_prediction(X,Y,steps=100):
    CO = causation(X,Y)
    D = {1:[],-1:[]}
    for i in xrange(steps):
        d = CO.ANM_predict_causality()
        D[d['causal_direction']].append(d['pvalscore'])

    #plt.hist([D[1],D[-1]],bins=30,label=['(X,Y)','(Y,X)'],alpha=0.5)
    #plt.legend()
    #plt.show()

    if len(D[1])> len(D[-1]):
        return 1
    else:
        return -1



def QA_single(steps=50,dataset_fn='pair0001.txt'):
    Z = np.loadtxt(dataset_fn)
    A, B = Z[:,0], Z[:,1]

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
