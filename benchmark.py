import os
from causation import *
import sys


def parse_ground_truth(fn='pair0001_des.txt'):
    for l in open(fn):
        if not l.startswith('ground truth:'):
            continue
        else:
            break
    cause = l.strip().split('-->')[0]
    ground_truth = 1 if cause=='x' else -1
    return ground_truth

def benchmark_pairs_data(datasetdir='./pairs_dataset'):
    datasets, ground_truths = dict(), dict()
    bad_datas = set()
    for root, dirs, filenames in os.walk(datasetdir):
        for f in filenames:
            key = f[:f.find('_')] if '_' in f else f[:f.find('.')]
            print os.path.join(datasetdir,f)
            if 'des' in f:
                gt = parse_ground_truth( os.path.join(datasetdir,f) )
                ground_truths[key] = gt
            else:
                fin = open( os.path.join(datasetdir,f) )
                A, B = [], []
                for l in fin:
                    l = l.strip().split()
                    if len(l) != 2:
                        bad_datas.add(f)
                        continue
                    a, b = l
                    if float(a) in A or float(b) in B: continue
                    A.append(float(a))
                    B.append(float(b))
                datasets[key] = (A,B)

    hit, nohit = 0., 0.
    for f in datasets:
        _a, _b = datasets[f]
        if len(_a) == 0:
            print 'Failed ', f
            continue
        print '\nProcessing ', f
        try:
            pred  = AMM_bagging_causality_prediction(_a,_b)['finalscore']
        except:
            print sys.exc_info()[0]
            print 'Failed ', f
            continue
        real = ground_truths[f]
        if pred*real > 0:
            hit += 1.
        else:
            nohit += 1.
        print 'Current acc:', hit, nohit, float(hit/(hit+nohit))
    acc = float(hit/(hit+nohit))
    return acc

def main():
    fac = benchmark_pairs_data()
    print '\nFinal Benchmark Accuracy:', fac

if __name__ == '__main__':
    main()
