import numpy as np
def neural_net(X=np.array([[1,0,0],[0,0,1],[1,0,0],[0,0,1]]),y=np.array([[1,0,1,0]]).T,syn0=(2*np.random.random((len(X[0]),1))-1)):
    for i in xrange(100000):
        l1 = 1/(1+np.exp(-(np.dot(X,syn0))))
        syn0 = np.dot(X.T,y-l1)
    return {'syn0':syn0,'l1':l1}
