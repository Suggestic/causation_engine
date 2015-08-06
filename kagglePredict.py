import os
from causation import *
import sys

def make_preds(fn_train='CEfinal_valid_pairs.csv'):
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

    fout = open('kagglePreds.csv','w',0)
    fout.write("SampleID,Target\n")
    for i in xrange(len(Xs)):
        pred  = AMM_bagging_causality_prediction(Xs[i],Ys[i])
        fout.write("%s,%s\n" % (names[i],pred))
        print "%s,%s\n" % (names[i],pred)


if __name__ == '__main__':
    make_preds()
