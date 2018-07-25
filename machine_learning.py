import tensorflow as tf

xData = [1, 2, 3, 4, 5, 6, 7]
yData = [25000, 55000, 75000, 110000, 128000, 155000, 180000]
# Rate
W = tf.Variable(tf.random_uniform([1], -100, 100))
# Bias
b = tf.Variable(tf.random_uniform([1], -100, 100))
# Build Place holders
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
# Hypothesis
H = W * X + b
# mean((H-Y)^2)
cost = tf.reduce_mean(tf.square(H - Y))
# Decent Gradient, Define how much it will jump
a = tf.Variable(0.01)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)
# Initialize variables
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(5001):
    sess.run(train, feed_dict={X: xData, Y: yData})
    if i % 500 == 0:
        print(i, sess.run(cost, feed_dict={X: xData, Y: yData}), sess.run(W), sess.run(b))
print(sess.run(H, feed_dict={X: [8]}))
 
