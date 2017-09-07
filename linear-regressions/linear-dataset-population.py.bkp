import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import xlrd
import io

DATA_FILE = "../data/ex1data1.xls"
# Step 1: read in data from the .xls file
book = xlrd.open_workbook(DATA_FILE, encoding_override="utf-8")
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(0, sheet.nrows)])
n_samples = sheet.nrows

plt.ion()
#fig, ax = plt.subplots(1, 1)
fig, ax = plt.subplots()
ax.set_xlabel('population of city')
ax.set_ylabel('profit of a food truck in city')
for x, y in data:
    ax.scatter(x, y)
fig.show()
plt.draw()
#fig.show()


# Step 2: create placeholders for input X (number of fire) and label Y (number of theft)
X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")
# Step 3: create weight and bias, initialized to 0
w = tf.Variable(0.0, name="weights")
b = tf.Variable(0.0, name="bias")
# Step 4: construct model to predict Y (number of theft) from the number of fire
Y_predicted = X * w + b
# Step 5: use the square error as the loss function
loss = tf.reduce_sum(tf.square(Y - Y_predicted), name="loss")


tf.summary.scalar("Cost", loss)

summary_op = tf.summary.merge_all()


# Step 6: using gradient descent with learning rate of 0.01 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
#writer = tf.summary.FileWriter("./dgraph", sess.graph)
with tf.Session() as sess:
    # Step 7: initialize the necessary variables, in this case, w and b
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("./dgraph", sess.graph)
    # Step 8: train the model
    for i in range(100): # run 100 epochs
        for x, y in data:
            # Session runs train_op to minimize loss
            #sess.run(optimizer, feed_dict={X: x, Y:y})
            _, summary = sess.run([optimizer, summary_op], feed_dict={X:x, Y:y})
        writer.add_summary(summary, i)    
    curr_W, curr_b, curr_loss  = sess.run([w, b, loss], {X:x, Y:y})
    for x, y in data:
        ax.plot(x, Y_predicted.eval(feed_dict={X: x}, session=sess), 'ro', lw=3)
        fig.show()
        plt.draw()
#writer = tf.summary.FileWriter("./dgraph", sess.graph)


print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
fig.show()
buf = io.BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)
image = tf.image.decode_png(buf.getvalue(), channels=4)
image = tf.expand_dims(image, 0)
summary_op = tf.summary.image("plot", image)
with tf.Session() as sess:
    # Run
    summary = sess.run(summary_op)
    # Write summary
    writer = tf.summary.FileWriter('./dgraph')
    writer.add_summary(summary)
    writer.close()

plt.waitforbuttonpress()
