import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def conv_simple(_input, _w, _b):
    # Reshape input
    _input_r = tf.reshape(_input, shape=[-1, 64, 64, 3])
    # Convolution
    _conv1 = tf.nn.conv2d(_input_r, _w['wc1'], strides=[1, 1, 1, 1], padding='SAME')
    # Add-bias
    _conv2 = tf.nn.bias_add(_conv1, _b['bc1'])
    # Pass ReLu
    _conv3 = tf.nn.relu(_conv2)
    # Max-pooling
    _pool  = tf.nn.max_pool(_conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # Vectorize
    _dense = tf.reshape(_pool, [-1, _w['wd1'].get_shape().as_list()[0]])
    # Fully-connected layer
    _out = tf.add(tf.matmul(_dense, _w['wd1']), _b['bd1'])
    # Return everything
    out = {
        'input_r': _input_r, 'conv1': _conv1, 'conv2': _conv2, 'conv3': _conv3
            , 'pool': _pool, 'dense': _dense, 'out': _out
        }
    print out
    return out
print ("CNN ready")

def main():
    cwd = os.getcwd()
    loadpath = cwd + "/data/custom_data.npz"
    l = np.load(loadpath)
    print (l.files)
    # Parse data
    trainimg = l['trainimg']
    trainlabel = l['trainlabel']
    print trainlabel.shape
    testimg = l['testimg']
    testlabel = l['testlabel']
    imgsize = l['imgsize']
    ntrain = trainimg.shape[0]
    nclass = trainlabel.shape[1]
    dim    = trainimg.shape[1]
    ntest  = testimg.shape[0]
    print ("%d train images loaded" % (ntrain))
    print ("%d test images loaded" % (ntest))
    print ("%d dimensional input" % (dim))
    print ("Image size is %s" % (imgsize))
    print ("%d classes" % (nclass))

    # Create the model
    x = tf.placeholder(tf.float32, [None, 12288])
    y = tf.placeholder(tf.float32, [None, 2])
    weights  = {
        'wc1': tf.Variable(tf.random_normal([3, 3, 3, 64], stddev=0.1)),
        'wd1': tf.Variable(tf.random_normal([32*32*64, 2], stddev=0.1))
    }
    biases   = {
        'bc1': tf.Variable(tf.random_normal([64], stddev=0.1)),
        'bd1': tf.Variable(tf.random_normal([2], stddev=0.1))
    }


    # Parameters
    learning_rate   = 0.001
    training_epochs = 100
    batch_size      = 5
    display_step    = 1
    # Functions! 
    _pred = conv_simple(x, weights, biases)['out']
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=_pred))
    optm = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    _corr = tf.equal(tf.argmax(_pred,1), tf.argmax(y,1)) # Count corrects
    accr = tf.reduce_mean(tf.cast(_corr, tf.float32)) # Accuracy
    init = tf.initialize_all_variables()
    # Saver 
    save_step = 1;
    savedir = "conv1/"
    saver = tf.train.Saver()
    print ("Network Ready to Go!")
    
    #sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess = tf.Session()
    writer = tf.summary.FileWriter("./graphs", sess.graph)
    sess.run(init)
    for epoch in range(training_epochs):
        avg_cost = 0.
        num_batch = int(ntrain/batch_size)
        # Loop over all batches
        for i in range(num_batch):
            randidx = np.random.randint(ntrain, size=batch_size)
            batch_xs = trainimg[randidx, :]
            batch_ys = trainlabel[randidx, :]
            sess.run(optm, feed_dict={x: batch_xs, y : batch_ys})
        # Display logs per epoch step

        if epoch % display_step == 0:
            print ("Epoch: %03d/%03d cost: %.9f" % (epoch, training_epochs, avg_cost))
            train_acc = sess.run(accr, feed_dict={x: batch_xs, y: batch_ys})
            print (" Training accuracy: %.3f" % (train_acc))
            test_acc = sess.run(accr, feed_dict={x: testimg, y: testlabel})
            print (" Test accuracy: %.3f" % (test_acc))

        # Save Net
        if epoch % save_step == 0:
            saver.save(sess, "conv1/cnn_simple.ckpt-" + str(epoch))
    print ("Optimization Finished.")


if __name__ == '__main__':
    sys.exit(main())
