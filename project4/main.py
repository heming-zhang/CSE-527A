import pickle
import numpy as np
import sys

# run this to save edit best_parameters.pickle which will be used to determine performance on the autograder
# Also feel free to use this file to do any testing as it will not be called by the autograder

transname = sys.argv[1]
round = int(sys.argv[2])
iter = int(sys.argv[3])
stepsize = float(sys.argv[4])
l2 = int(sys.argv[5])
l3 = int(sys.argv[6])
l4 = int(sys.argv[7])
print(l2, l3, l4)
print("parameters:",transname,round,iter,stepsize,np.array([1,l2,l3,l4,13]))

best_parameters = {
    'TRANSNAME' : transname,
    'ROUNDS' : round,
    'ITER' : iter,
    'STEPSIZE' : stepsize,
    'wst' : np.array([1,l2,l3,l4,13])

    # 'TRANSNAME' : 'sigmoid',
    # 'ROUNDS' : 50,
    # 'ITER' : 10,
    # 'STEPSIZE' : 0.05,
    # 'wst' : np.array([1,10,20,30,13])
}

with open('best_parameters.pickle', 'wb') as f:
    pickle.dump(best_parameters, f)