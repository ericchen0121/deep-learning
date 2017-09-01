# TensorFlow in 5 Minutes
# https://www.youtube.com/watch?v=2FmcHiLCwTU

# This video is all about building a handwritten digit image classifier in
# Python in under 40 lines of code (not including spaces and comments).
# We'll use the popular library TensorFlow to do this.

# ---------import MNIST data
#
import input_data
mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)

import tensorflow as tf

# ---------Set parameters
#
# If our learning rate is too big, our model may skip the optimal solution
# If it is too small, we may need too many iterations to CONVERGE onto the best
# result.
# 0.01 is a known decent learning rate for this problem.
learning_rate = 0.0075
training_iteration = 100
batch_size = 100
display_step = 5

# Create our model.
# In tensorflow, a model is represented as a data-flow graph.
# The graph contains a set of nodes, called OPERATIONS. These are units of
# computation, as simple as addition or subtraction. Or, as complicated as a
# multi-variate equation.
# Each operation takes in as INPUT a tensor, and outputs a tensor as wellself.
# Tensors are multidimensional arrays of numbers, and they flow between operations.
# Thus, the name TensorFlow.

# ---------Create placeholders.
#
# TF graph input
# We start building our model with two operations. Both are PLACEHOLDER OPERATIONS.
# A PLACEHOLDER is just a variable we'll assign data to at a later date.
# It's never initialized and contains no data.
# We'll define the type ('float') and shape (ie. 784) of our data as parameters.

# The input image 'x', will be represented by a 2D tensor of numbers.
# 784 is the dimensionality of a single flattened MNIST image file.
# FLATTENING an image means converting a 2D array to a 1D array, by unstacking
# the rows and lining them up. This is more efficient formatting.
x = tf.placeholder('float', [None, 784]) # mnist data image of shape 28*28=784

# The output 'y' will consist of a 2D tensor, where each row is a one hot 10
# dimensional vector showing which digit class the MNIST image belongs to.
y = tf.placeholder('float', [None, 10]) #0-9 digits recogition => 10 classes

# ---------Create a model.
#
# Set model weights.
# The WEIGHTS are the probabilities that will affect how the
# data flows in the graph, and they'll be updated continuously during training
# so our results get closer and closer to the right solution.
# The BIAS lets us shift our regression line to better fit the data.
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# ----------Create name scopes.
#
# SCOPES help us organize nodes in the graph visualizer called Tensor Board.
# We'll create 3 scopes.

# Scope 1
with tf.name_scope('Wx_b') as scope:
    # Construct a linear model.
    # In our first scope, we'll implement our model - logistic regression - by
    # matrix multiplying the input images 'x' by the weight 'W', and adding the
    # bias 'b'
    model = tf.nn.softmax(tf.matmul(x, W) + b)

# Add summary operations to collect data
# These summary operations help us later visualize the distribution of our
# weights and biases.
w_h = tf.summary.histogram('weights', W)
b_h = tf.summary.histogram('biases', b)

# These other name scopes will clean up graph representation

# Scope 2: The cost function.
# The COST FUNCTION helps us minimize our error during training.
with tf.name_scope('cost_function') as scope:
    # Minimize error using the popular cross entropy function.
    cost_function = -tf.reduce_sum(y*tf.log(model))

    # create a summary to monitor and visualize the cost function
    tf.summary.scalar('cost_function', cost_function)

# Scope 3: Train
# This will create our optimization function which will make our model improve
# during training. We'll use the popular Gradient Descent algorithm, which will
# take the learning rate for pacing
with tf.name_scope('train') as scope:
    # Gradient descent optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

# Initializing the variables
init = tf.global_variables_initializer()

# Merge all our summaries into a single operator (because we're lazy!)
merged_summary_op = tf.summary.merge_all()

# https://stackoverflow.com/questions/36666331/learning-rate-larger-than-0-001-results-in-error
# merged_summary_op = tf.constant(1)

# ---------Launch our Graph
# Now we initialize our session which allows us to execute our data-flow Graph
with tf.Session() as sess:
    sess.run(init)

    # Sets the log writer to a folder on your hard drive
    # set the summarywriter location, where we'll later load data from to
    # visualize in TensorBoard
    # summary_writer = tf.summary.FileWriter('/tmp/logs', graph_def=sess.graph_def)
    summary_writer = tf.summary.FileWriter('/tmp/logs', graph=sess.graph)
    # Training cycle!
    # setting our for loop for specified number of iterations.
    for iteration in range(training_iteration):
        # initialize our average cost, and print out to make sure our model is improving
        avg_cost = 0.
        # compute our batch size
        total_batch = int(mnist.train.num_examples/batch_size)

        # start training over each example in our training data
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)

            # Fit our model using our batch data and the gradient descent algorithm
            # for backpropagation
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})

            # Compute the average loss
            avg_cost += sess.run(cost_function, feed_dict={x: batch_xs, y: batch_ys})/total_batch

            # Write logs for each iteration via the SummaryWriter
            summary_str = sess.run(merged_summary_op, feed_dict={x: batch_xs, y: batch_ys})
            summary_writer.add_summary(summary_str, iteration*total_batch + i)

        # Display logs per iteration step
        if iteration % display_step == 0:
            print 'Iteration:', '%04d' % (iteration + 1), 'cost=', '{:.9f}'.format(avg_cost)

    print 'Tuning / training completed!'

    # Test the model
    prediction = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))

    # Calculate the accuracy
    accuracy = tf.reduce_mean(tf.cast(prediction, 'float'))
    print 'Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels})
