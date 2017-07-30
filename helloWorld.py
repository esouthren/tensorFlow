import tensorflow as tf

print "Hello World!"

'''
the computational graph is a series of TF operations arranged into a graph of nodes. Each nodes takes 0+ tensors as inputs and outputs a tensor. 

Node 1/2 are floating point constants - takes no imputs, simply outputs a value it stores internally.
'''

node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # also type float 32, implicitly)

# print(node1, node2)

''' 
when printed the nodes do not display their value - this happens when the computational graph is is run within a session.

Creating and running a session: 

'''
sesh = tf.Session()
print(sesh.run([node1, node2]))

'''
Combining nodes with operations (operations are nodes too). Adding two nodes and making a graph:
'''

node3 = tf.add(node1, node2)
print("node3: ", node3)
print("sesh.run(node3): ", sesh.run(node3))

'''
a placeholder is a promise to provide a value... later.
'''
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b
# shortcut for tf.add(a, b)

'''
testing the placeholder
'''

print(sesh.run(adder_node, {a: 3, b: 2}))
print(sesh.run(adder_node, {a:[1,4], b: [2,4]}))

'''
How about a more complex operation?
'''

add_and_triple = adder_node * 3
print(sesh.run(add_and_triple, {a: 3, b: 4.5}))

'''
how do we start training the network?

We need to get new outputs from the same inputs. 

Variables allow us to add trainable paramters. They have a type and initial value.
'''

w = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32) # we'll add the value x later, it's currently a promise
linear_model = w * x + b

'''
variables aren't initialised upon creation. We need to use this operation:
'''

init = tf.global_variables_initializer()
sesh.run(init)

'''
we can evaluate linear_model with several values of x if we feed it an array: '''

print(sesh.run(linear_model, {x:[1,2,3,4]}))

'''
how good is this model? we have no indicator. 
We create a y placeholder to provide the desired values, and we need a loss function. 

Loss function = how far away is the current model from the provided data?

A standard loss model for linear regression sums the squares of the deltas (difference) between current model and provided data (y) '''

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sesh.run(loss, {x:[1,2,3,4], y:[0,-1,-2, -3]}))

'''
This creates a loss value of 23.66

To get a loss value of 0, the value of w=-1 and b=1. We can change the values of the variables like so: 
'''
fixedw = tf.assign(w, [-1])
fixedb = tf.assign(b, [1])
sesh.run([fixedw, fixedb])
print(sesh.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

''' 
We guessed the values of w and b. 

The point of machine learning is to find these values automatically!

Tensor Flow provides Optimisers that slowly change each variable to minimise the loss function. 

The simplest is Gradient Descent. Variables are modified according to the magnitude of the loss to that variable. '''

optimiser = tf.train.GradientDescentOptimizer(0.01)
train = optimiser.minimize(loss)

sesh.run(init) # reset w and b
for i in range(1000):
    sesh.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})
    
print(sesh.run([w,b]))

'''
this gives us the values -0.99999 and 0.99999: the optimised values of b and w.'''

curr_w, curr_b, curr_loss = sesh.run([w,b, loss], {x:[1,2,3,4], y:[-1,-2,-3,-4]})
print("W: %s b: %s loss: %s"%(curr_w, curr_b, curr_loss))

