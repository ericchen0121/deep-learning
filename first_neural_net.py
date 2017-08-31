# Siraj Raval: Build a Neural Net in 4 Minutes
# https://www.youtube.com/watch?v=h3l4qz76JhQ

import numpy as np

# Sigmoid, a function that will map any value to a value between zero and one
# will be run at every neuron of our network when data hits it
# useful for creating probabilities out of numbers
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)

    return 1/(1+np.exp(-x))

# input data as a matrix
# each row is a different training example
# each column represents a different neuron
# so we have four training examples with three input neurons each
X = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])

# Create output data set
# output data set, which is four examples, with one output neuron each
y= np.array([[0],
            [1],
            [1],
            [0]])

# Seed the Random Numbers.
# We'll be generating random numbers soon.
# So here we'll seed those numbers to be deterministic.
# This means giving the randomly generated numbers the same starting point
# so we'll get the same sequence of generated numbers every time we run our
# program. NOTE: Seeding like this is is useful for debugging.
np.random.seed(1)

# Create synapse matricies
# Synapses are connections between each neuron in one layer to every neuron in
# the next layer.
# Since we'll have three layers in our neural network, we need two synapse matricies.

# Each synampse has a random weight given to it.
syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1

# Training step
# NOTE: in python3, this is: for j in range(100000)
for j in xrange(100000):

    # first layer is just the input data
    l0 = X

    # prediction step
    # here we perform matrix multiplication between each layer and its synapse
    # then we run the sygmoid function on all the values in the matrix to create
    # the next layer.
    l1 = nonlin(np.dot(l0, syn0))

    # This layer contains the prediction of the output data
    # We do th e same stypes on this layer to get the next layer.
    # This is a more refined prediction
    l2 = nonlin(np.dot(l1, syn1))

    # Now that we have a prediction of the output value in layer two
    # Let's compare it to the expected output data using subtraction to get the
    # error rate.
    l2_error = y - l2

    # Print out the average error rate at a set interval to make sure it goes
    # down every time!
    if(j % 10000) == 0:
        print "Error:" + str(np.mean(np.abs(l2_error)))

    # Next, we multiply the error rate by the result of the sigmoid function.
    # The function is used to get the derivative of our output prediction from
    # layer two. This will give us a delta, which we'll use to reduce the error
    # rate  of our predictions when we update our synapses EVERY ITERATION
    l2_delta = l2_error * nonlin(l2, deriv=True)

    # Then we'll want to see how much layer one contributed to the error in
    # layer two. This is called BACKPROPAGATION. We'll get this error by
    # multiplying layer two's delta by synapse one's transpose
    l1_error = l2_delta.dot(syn1.T)

    # Then we'll get synapse one's delta by multiplying layer one's error by
    # by the result of our sigmoid function. The function is used to get the
    # derivative of layer one.
    l1_delta = l1_error * nonlin(l1, deriv=True)

    # Update Weights
    # Now we have deltas for each of our layers, we can use them to update our
    # synapse weights to reduce our error weight MORE AND MORE each iteration.
    # This is an algorithm called GRADIENT DESCENT.
    # To do this, we'll just multiply each layer by a delta.
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

print "Output after training"
print l2
