import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.misc import imread, imresize


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
    
    image = imread(sys.argv[1])
    imgsize = [64, 64]
    print (image.shape)
    small_image = imresize(image, [imgsize[0], imgsize[1]])/255.
    print (small_image.shape)
    vec_image = np.reshape(small_image, (1, -1))
    print (vec_image.shape)

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
    # Functions! 
    _pred = conv_simple(x, weights, biases)['out']
    _corr = tf.argmax(_pred,1) # Count corrects
    # Saver 
    saver = tf.train.Saver()
    saver = tf.train.import_meta_graph("conv1/cnn_simple.ckpt-99.meta")
    print ("NETWORK RESTORED")
    with tf.Session() as sess:
        saver.restore(sess, "conv1/cnn_simple.ckpt-99")
        print(sess.run(weights))
        if (sess.run(_corr, feed_dict={x: vec_image})):
            print ("It is a Bird")
        else:
            print ("It is not a Bird")


if __name__ == '__main__':
    sys.exit(main())
