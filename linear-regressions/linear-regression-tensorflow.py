import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import xlrd
import io
import shutil

# Deleting the logdir
try:
   shutil.rmtree("dgraph/")
except:
   pass
# Read the Data set

DATA_FILE = "../data/ex1data1.xls"
book = xlrd.open_workbook(DATA_FILE, encoding_override="utf-8")
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(0, sheet.nrows)])
n_samples = sheet.nrows


def normalize(array):
    return (array - array.mean()) / array.std()
# Normalize a data set

size_data_n = normalize(data[:, [0]])

# Plotting the data set
fig, ax = plt.subplots()
ax.set_xlabel('population of city')
ax.set_ylabel('profit of a food truck in city')
plt.plot(data[:, [0]], data[:, [1]], 'ro', label='Samples data')

# Creating intial place holders X and Y
X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

# Creating  variables for Weight and bias 
# and initialized to 0.0

w = tf.Variable(0.0, name="weights")
b = tf.Variable(0.0, name="bias")


# linear Model
Y_predicted = X * w + b


# Caliculating the loss 
loss = tf.reduce_sum(tf.square(Y - Y_predicted), name="loss")

# Using gradient descent to minmize the loss

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

# Adding the loss to the   tensorboard scalar
tf.summary.scalar("Cost", loss)
summary_op = tf.summary.merge_all()

#Initializing the tensorflow Session

with tf.Session() as sess:
    # Initialize variables in tensorflow 
    sess.run(tf.global_variables_initializer())
    # Initialize the writer to write values for tensorboard
    writer = tf.summary.FileWriter("./dgraph", sess.graph)

    #  Training the model
    for i in range(100): # run 100 epochs/iterations
        for x, y in data:
            # Session runs train_op to minimize loss
            _, summary = sess.run([optimizer, summary_op], feed_dict={X:x, Y:y})
        # writing the loss to scalar graph
        print i
        writer.add_summary(summary, i)

    # optimized weight , bias and caliculated loss for the given model

    curr_W, curr_b, curr_loss  = sess.run([w, b, loss], {X:x, Y:y})

    #  plotting the predicted output
    plt.plot(data[:, [0]], curr_W * data[:, [0]] + curr_b, label='Fitted line')
    plt.legend()
    print("W: %s b: %s loss: %s" %(curr_W, curr_b, curr_loss))

    # Generating plot and png image to export into tensorboard
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    summary_op = tf.summary.image("plot", image)

    # Writing the image to tensorboard 
    summary_plot = sess.run(summary_op)
    writer.add_summary(summary_plot)
    writer.close()
    
    print  "#########################################################"
    print  "##### Estimated  profit for the population 35000  #######"
    print  -((curr_W*3.5)+curr_b)*10000
plt.waitforbuttonpress()
