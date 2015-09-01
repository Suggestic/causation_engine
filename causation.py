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
from sklearn.metrics.pairwise import pairwise_kernels
import warnings
from scipy.stats import mannwhitneyu
import pyGPs


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

    def ANM_predict_causality(self,train_size=0.5,independence_criterion='HSIC',metric='linear'):
        '''
            Prediction of causality based on the bivariate additive noise model

            Parameters
            ----------
            independence_criterion :
                kruskal for Kruskal-Wallis H-test,
                HSIC for Hilbert-Schmidt Independence Criterion

            Returns
            -------
            Causal-direction: 1 if X causes Y, or -1 if Y causes X
        '''
        Xtrain, Xtest , Ytrain, Ytest = train_test_split(self.X, self.Y, train_size = train_size)
        _gp = KernelRidge(kernel='rbf',degree=3)#GaussianProcess()#

        #Forward case
        _gp.fit(Xtrain,Ytrain)
        errors_forward = _gp.predict(Xtest) - Ytest

        #Backward case
        _gp.fit(Ytrain,Xtrain)
        errors_backward = _gp.predict(Ytest) - Xtest

        #Independence score

        forward_indep_pval = {
            'kruskal': kruskal(errors_forward,Xtest)[1],
            'HSIC': self.HilbertSchmidtNormIC(errors_forward,Xtest)[1]
        }[independence_criterion]

        backward_indep_pval = {
            'kruskal': kruskal(errors_backward,Ytest)[1],
            'HSIC': self.HilbertSchmidtNormIC(errors_backward,Ytest)[1]
        }[independence_criterion]

        #print 'Scores:', forward_indep_pval, backward_indep_pval

        if forward_indep_pval < backward_indep_pval:
            self.causal_direction = 1
            self.pvalscore = forward_indep_pval
        else:
            self.causal_direction = -1
            self.pvalscore = backward_indep_pval

        return {'causal_direction':self.causal_direction,'pvalscore':self.pvalscore,'difways':abs(forward_indep_pval-backward_indep_pval)}



    def ANM_causation_score(self,train_size=0.5,independence_criterion='HSIC',metric='sigmoid',regression_method='GP'):
        '''
            Measure how likely a given causal direction is true

            Parameters
            ----------
            train_size :
                Fraction of given data used to training phase

            independence_criterion :
                kruskal for Kruskal-Wallis H-test,
                HSIC for Hilbert-Schmidt Independence Criterion

            metric :
                linear, sigmoid, rbf, poly
                kernel function to compute gramm matrix for HSIC
                gaussian kernel is used in :
                Nonlinear causal discovery with additive noise models
                Patrik O. Hoyer et. al

            Returns
            -------
            causal_strength: A float between 0. and 1.
        '''
        Xtrain, Xtest , Ytrain, Ytest = train_test_split(self.X, self.Y, train_size = train_size)
        if regression_method == 'GP':
            _gp = GaussianProcess(regr='quadratic')#KernelRidge(kernel='sigmoid',degree=3)

            _gp = pyGPs.GPR()      # specify model (GP regression)
            _gp.getPosterior(Xtrain, Ytrain) # fit default model (mean zero & rbf kernel) with data
            _gp.optimize(Xtrain, Ytrain)     # optimize hyperparamters (default optimizer: single run minimize)

            #Forward case
            #_gp = KernelRidge(kernel='sigmoid',degree=3)
            #_gp.fit(Xtrain,Ytrain)
            ym, ys2, fm, fs2, lp = _gp.predict(Xtest)
            _gp.plot()
            #errors_forward = _gp.predict(Xtest) - Ytest
            errors_forward = ym - Ytest
        else:
            _gp = KernelRidge(kernel='sigmoid')
            _gp.fit(Xtrain, Ytrain)
            errors_forward = _gp.predict(Xtest) - Ytest

        #Independence score

        forward_indep_pval = {
            'kruskal': kruskal(errors_forward,Xtest)[1],
            'HSIC': self.HilbertSchmidtNormIC(errors_forward,Xtest)[1]
        }[independence_criterion]


        return {'causal_strength':forward_indep_pval}





    def HilbertSchmidtNormIC(self,X,Y,metric='linear'):
        '''
            Procedure to calculate the Hilbert-Schmidt Independence Criterion described in
            "Measuring Statistical Dependence with Hilbert-Schmidt Norms", Arthur Gretton et al.

            Parameters
            ----------
            Assuming a joint distribution P(X,Y)
            X :
                list of X observations
            Y : list of Y obervations

            Returns
            -------
            (HSIC, fake-p-value scaling HSIC to [0,1])
        '''
        m = float(len(X))
        K = pairwise_kernels(X,X,metric=metric)
        L = pairwise_kernels(Y,Y,metric=metric)
        H = np.eye(m)-1/m

        res = (1/(m-1)**2 ) * np.trace(np.dot(np.dot(np.dot(K,H),L),H))

        #Another way, maybe..
        #CCm = pairwise_kernels(self.X,self.Y)
        #res = sum(np.linalg.eigvals(CCm))

        #return (res,1-(1/(1+res) ) )
        return (res,res )

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


    def plot_data_densities(self,bw_method='scott'):
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
        positionsx = np.vstack([X.ravel()])
        positionsy = np.vstack([Y.ravel()])

        values = np.vstack([m1, m2])
        valuesx = m1
        valuesy = m2

        kernel = stats.gaussian_kde(values,bw_method=bw_method)
        kernelx = stats.gaussian_kde(valuesx,bw_method=bw_method)
        kernely = stats.gaussian_kde(valuesy,bw_method=bw_method)

        Z = np.reshape(kernel(positions).T, X.shape)
        _x = np.reshape(kernelx(positionsx).T, X.shape)
        _y = np.reshape(kernely(positionsy).T, X.shape)
        Zx = Z/_y
        Zy = Z/_x
        Zw = _x*_y
        #fig = plt.figure(figsize=(12, 6))
        f, ((ax1, ax2, ax5), (ax3, ax4, ax6)) = plt.subplots(2, 3, sharex='col', sharey='row',figsize=(17, 10))
        f.subplots_adjust(hspace=0.1)
        f.subplots_adjust(wspace=0.1)

        #ax = fig.add_subplot(221)
        ax1.imshow(np.rot90(Z), cmap=plt.cm.RdYlBu,
                   extent=[xmin, xmax, ymin, ymax])
        ax1.plot(m1, m2, 'k.', markersize=5,linewidth='0.1',color='b',alpha=0.4)

        #ax = fig.add_subplot(222)
        ax2.imshow(np.rot90(Zx), cmap=plt.cm.RdYlBu,
                   extent=[xmin, xmax, ymin, ymax])
        ax2.plot(m1, m2, 'k.', markersize=5,linewidth='0.1',color='b',alpha=0.4)

        #ax = fig.add_subplot(223)
        ax3.imshow(np.rot90(Zy), cmap=plt.cm.RdYlBu,
                   extent=[xmin, xmax, ymin, ymax])
        ax3.plot(m1, m2, 'k.', markersize=5,linewidth='0.1',color='b',alpha=0.4)

        ax4.imshow(np.rot90(Zw), cmap=plt.cm.RdYlBu,
                   extent=[xmin, xmax, ymin, ymax])
        ax4.plot(m1, m2, 'k.', markersize=5,linewidth='0.1',color='b',alpha=0.4)

        ax5.imshow(np.rot90(_x), cmap=plt.cm.RdYlBu,
                   extent=[xmin, xmax, ymin, ymax])
        ax5.plot(m1, m2, 'k.', markersize=5,linewidth='0.1',color='b',alpha=0.4)

        ax6.imshow(np.rot90(_y), cmap=plt.cm.RdYlBu,
                   extent=[xmin, xmax, ymin, ymax])
        ax6.plot(m1, m2, 'k.', markersize=5,linewidth='0.1',color='b',alpha=0.4)

        ax1.set_title('P(X,Y)')
        ax2.set_title('P(X|Y)')
        ax3.set_title('P(Y|X)')
        ax4.set_title('P(X)*P(Y)')
        ax5.set_title('P(X)')
        ax6.set_title('P(Y)')



        ax1.set_xlim([xmin, xmax])
        ax1.set_ylim([ymin, ymax])
        ax2.set_xlim([xmin, xmax])
        ax2.set_ylim([ymin, ymax])
        ax3.set_xlim([xmin, xmax])
        ax3.set_ylim([ymin, ymax])
        ax4.set_xlim([xmin, xmax])
        ax4.set_ylim([ymin, ymax])
        ax5.set_xlim([xmin, xmax])
        ax5.set_ylim([ymin, ymax])
        ax6.set_xlim([xmin, xmax])
        ax6.set_ylim([ymin, ymax])

        ax1.axis('off')
        ax2.axis('off')
        ax3.axis('off')
        ax4.axis('off')
        ax5.axis('off')
        ax6.axis('off')

        self.forceAspect(ax1,aspect=1)
        self.forceAspect(ax2,aspect=1)
        self.forceAspect(ax3,aspect=1)
        self.forceAspect(ax4,aspect=1)
        self.forceAspect(ax5,aspect=1)
        self.forceAspect(ax6,aspect=1)


        plt.setp([a.get_xticklabels() for a in [ax3,ax4,ax6]], visible=False)
        plt.setp([a.get_yticklabels() for a in [ax1,ax3,ax6]], visible=False)

        plt.show()

        return None

def AMM_bagging_causality_prediction(X,Y,steps=26,independence_criterion='HSIC',metric='linear'):
    CO = causation(X,Y)
    D = {1:[],-1:[]}
    _D = {1:[],-1:[]}

    for i in xrange(steps):
        d = CO.ANM_predict_causality(independence_criterion=independence_criterion,metric=metric)
        D[d['causal_direction']].append(d['pvalscore'])
        _D[d['causal_direction']].append(d['difways'])


    #plt.hist([D[1],D[-1]],bins=30,label=['(X,Y)','(Y,X)'],alpha=0.5)
    #plt.legend()
    #plt.show()

    meanxy, meanyx = np.mean(_D[1]), np.mean(_D[-1])
    sumxy, sumyx = sum(_D[1]), sum(_D[-1])
    meanxy = 0 if np.isnan(meanxy) else meanxy
    meanyx = 0 if np.isnan(meanyx) else meanyx
    mwu = mannwhitneyu(_D[1],_D[-1])

    if sumxy > sumyx:
        return {'causal_direction':1,'finalscore': 1/(1+mwu[1])}
    else:
        return {'causal_direction':-1,'finalscore': -1/(1+mwu[1])}



def QA_single(steps=50,dataset_fn='pair0001.txt'):
    Z = np.loadtxt(dataset_fn)
    A, B = Z[:,0], Z[:,1]

    plt.scatter(A,B)
    plt.show()

    CO = causation(A,B)
    D = {1:[],-1:[]}
    _D = {1:[],-1:[]}

    for i in xrange(steps):
        d = CO.ANM_predict_causality()
        D[d['causal_direction']].append(d['pvalscore'])
        _D[d['causal_direction']].append(d['difways'])

    #plt.hist([D[1],D[-1]],bins=40,label=['(X,Y)','(Y,X)'],alpha=0.5,histtype='stepfilled')
    plt.hist([_D[1],_D[-1]],bins=40,label=['(X,Y)','(Y,X)'],alpha=0.5,histtype='stepfilled')

    plt.legend()
    plt.show()

    meanxy, meanyx = np.mean(_D[1]), np.mean(_D[-1])
    sumxy, sumyx = sum(_D[1]), sum(_D[-1])
    meanxy = 0 if np.isnan(meanxy) else meanxy
    meanyx = 0 if np.isnan(meanyx) else meanyx

    print '#X->Y:', len(_D[1])
    print '#Y->X:', len(_D[-1])
    print 'mean X->Y:', meanxy
    print 'mean Y->X:', meanyx
    print 'sum X->Y:', sumxy
    print 'sum Y->X:', sumyx
    mwu = mannwhitneyu(_D[1],_D[-1])
    print 'mannwhitneyu:', mwu


    if sumxy > sumyx:
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

def HSICtest():
    dataset_fn='pair0001.txt'
    Z = np.loadtxt(dataset_fn)
    A, B = Z[:,0], Z[:,1]
    print 'cause (A,B)?:', causation(A,B).ANM_predict_causality()
    print 'cause (B,A)?:', causation(B,A).ANM_predict_causality()

if __name__ == '__main__':
    #main()
    #if QA_single() == 1:
    #    print '\n\n X --> Y'
    #else:
    #    print '\n\n Y --> X'
    HSICtest()
