import tensorflow as tf

print "Hello World!"

'''
the computational graph is a series of TF operations arranged into a graph of nodes. Each nodes takes 0+ tensors as inputs and outputs a tensor. 

Node 1/2 are floating point constants - takes no imputs, simply outputs a value it stores internally.
'''

node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # also type float 32, implicitly)

print(node1, node2)

