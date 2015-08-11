import os
from causation import *
import sys
from multiprocessing import Pool
from threading import BoundedSemaphore
from random import randint
import shelve


pool_sema = BoundedSemaphore(value=1)

def ANM_wrap(D):
    name,X, Y = D[0], D[1], D[2]

    pool_sema.acquire()
    dictres = shelve.open('shelveKaggleResDict')

    if name in dictres:
        dictres.close()
        print 'Skipping ', name
        return 1

    dictres.close()
    pool_sema.release()

    res = AMM_bagging_causality_prediction(X,Y)['finalscore']

    pool_sema.acquire()
    dictres = shelve.open('shelveKaggleResDict')
    dictres[name] = res
    print "\n%s,%s" % (name,res)
    print 'Already processed: ', len(dictres)
    #print dictres
    dictres.close()
    pool_sema.release()
    return 0

def make_datasets(fn_train='CEfinal_valid_pairs.csv'):
    fin = open(fn_train,'r')
    fin.readline()
    names, Xs, Ys = [], [], []
    for l in fin:
        l = l.strip().split(',')
        name = l[0]
        X = [float(x) for x in l[1].split()]
        Y = [float(y) for y in l[2].split()]
        names.append(name)
        Xs.append(X)
        Ys.append(Y)

    return {'names':names,'Xs':Xs,'Ys':Ys}

def main():

    dss = make_datasets()
    names, Xs, Ys = dss['names'], dss['Xs'], dss['Ys']

    fout = open('kagglePreds.csv','w',0)
    fout.write("SampleID,Target\n")

    args = [x for x in zip(names,Xs,Ys)]

    p = Pool(4)#randint(3,4))
    p.map( ANM_wrap, args )
    p.close()
    p.join()

    dictres = shelve.open('shelveKaggleResDict')


    for i in xrange(len(Xs)):
        fout.write("%s,%s\n" % (names[i],dictres[names[i]]))



if __name__ == '__main__':
    main()
