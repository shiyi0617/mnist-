import sys, os
sys.path.append(os.pardir)
from neuralnet_mnist import get_data,init_network,predict
import numpy as np
x,_=get_data()
network=init_network()
W1,W2,W3=network['W1'],network['W2'],network['W3']
print(x.shape)
print(x[0].shape)
print(W1.shape,W2.shape,W3.shape)

