'''
Manuscript Associated: New developments and comparisons of manifold-based surrogates with deep neural operators
Authors: Katiana Kontolati, PhD Candidate, Johns Hopkins University
         Somdatta Goswami, Postdoctoral Researcher, Brown University
Tensorflow Version Required: TF1.15     
This should be used for sharp data    

Before running the code: Provide the path for the training and testing dataset in utils/dataset.py Line: 12
                         Provide the path for the out-of-distribution dataset in utils/dataset.py Line: 13
                         
Last update: February 5, 2022
'''

import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import numpy as np
import time

from utils.fnn import FNN
from utils.conv import CNN
from utils.dataset import DataSet_POD as DataSet
from utils.savedata import SaveData
from utils.savedataOOD import SaveDataOOD
from utils.savedatanoisy import SaveDataNoisy

print("You are using TensorFlow version", tf.__version__)
np.random.seed(1234)

#output dimension of Branch/Trunk
p = 100
modes = 180 # 100
#fnn in CNN
layer_B = [256, modes]
#trunk net
layer_T = [2, 128, 128, 128, p]

#resolution
h = 28
w = 28

#parameters in CNN
n_channels = 1
n_out_channels = 16
filter_size_1 = 8
filter_size_2 = 8
filter_size_3 = 8
filter_size_4 = 8
stride = 1

#filter size for each convolutional layer
num_filters_1 = 16
num_filters_2 = 16
num_filters_3 = 16
num_filters_4 = 16

#batch_size
bs = 200

#size of input for Trunk net
nx = 28*28*10
x_num = nx
num = 28*28
def main():
        
    data = DataSet(nx, bs, modes)
    x_train, f_train, u_train, Xmin, Xmax = data.minibatch()
    u_basis = data.PODbasis()
    
    x_pos = tf.constant(x_train, dtype=tf.float32)
    x = tf.tile(x_pos[None, :, :], [bs, 1, 1]) #[bs, x_num, x_dim]

    f_ph = tf.placeholder(shape=[None, h, w, n_channels], dtype=tf.float32) #[bs, 1, h, w, n_channels]
    u_ph = tf.placeholder(shape=[None, x_num, 1], dtype=tf.float32) #[bs, x_num, 1]
    
    conv_model = CNN()
    conv_1 = conv_model.conv_layer(f_ph, filter_size_1, num_filters_1, stride, actn=tf.nn.relu)
    pool_1 = conv_model.avg_pool(conv_1, ksize=2, stride=2) 
    conv_2 = conv_model.conv_layer(pool_1, filter_size_2, num_filters_2, stride, actn=tf.nn.relu)
    pool_2 = conv_model.avg_pool(conv_2, ksize=2, stride=2) 
    conv_3 = conv_model.conv_layer(pool_2, filter_size_3, num_filters_3, stride, actn=tf.nn.relu)
    pool_3 = conv_model.avg_pool(conv_3, ksize=2, stride=2)
    conv_4 = conv_model.conv_layer(pool_3, filter_size_4, num_filters_4, stride, actn=tf.nn.relu)
    pool_4 = conv_model.avg_pool(conv_4, ksize=2, stride=2)  
    layer_flat = conv_model.flatten_layer(pool_4)

    fnn_layer_1 = conv_model.fnn_layer(layer_flat, layer_B[0], actn=tf.tanh, use_actn=True)
    out_B = conv_model.fnn_layer(fnn_layer_1, layer_B[1], actn=tf.tanh, use_actn=False) #[bs, p]

    #POD basis
    u_basis = tf.constant(u_basis, dtype=tf.float32)

    #prediction
    u_pred = tf.einsum('bi,ni->bn', out_B, u_basis)
    u_pred = tf.tile(u_pred[:, :, None], [1, 1, 1])

    loss = tf.reduce_mean(tf.square(u_ph - u_pred)) + 10*tf.reduce_mean(tf.square(u_ph[:,6*num:7*num,:] - u_pred[:,6*num:7*num,:])) + \
           10*tf.reduce_mean(tf.square(u_ph[:,7*num:8*num,:] - u_pred[:,7*num:8*num,:])) + 10*tf.reduce_mean(tf.square(u_ph[:,0:num,:] - u_pred[:,0:num,:]))
    train = tf.train.AdamOptimizer(learning_rate=1.0e-3, beta1 = 0.99).minimize(loss)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
    sess.run(tf.global_variables_initializer())

    n = 0
    nmax = 1500

    train_loss = np.zeros((nmax+1, 1))
    test_loss = np.zeros((nmax+1, 1))

    start_time = time.perf_counter()
    time_step_0 = time.perf_counter()
    while n <= nmax:

        x_train, f_train, u_train, _, _ = data.minibatch()
        train_dict={f_ph: f_train, u_ph: u_train}
        loss_, _ = sess.run([loss, train], feed_dict=train_dict)

        if n%1 == 0:
            test_id, x_test, f_test, u_test = data.testbatch(bs)
            u_test_ = sess.run(u_pred, feed_dict={f_ph: f_test})
            u_test = data.decoder(u_test)
            u_test_ = data.decoder(u_test_)
            err = np.mean(np.linalg.norm(u_test_ - u_test, 2, axis=1)/np.linalg.norm(u_test, 2, axis=1))
            time_step_1000 = time.perf_counter()
            T = time_step_1000 - time_step_0
            print('Step: %d, Loss: %.3e, Test L2 error: %.3e, Time (secs): %.3f'%(n, loss_, err, T))
            time_step_0 = time.perf_counter()
        
        train_loss[n,0] = loss_
        test_loss[n,0] = err
        n += 1

    stop_time = time.perf_counter()
    print('Elapsed time (secs): %.3f'%(stop_time - start_time))
    
    # Save results for test data
    data_save = SaveData()
    num_test = 100
    data_save.save_POD(sess, x_pos, f_ph, u_ph, u_pred, data, num_test, h)

    # Save results for OOD data
    data_save_ood = SaveDataOOD()
    num_ood = 100
    data_save_ood.save_POD(sess, x_pos, f_ph, u_ph, u_pred, data, num_ood, h)

    # Save results for noisy data
    num_noisy = 200
    noise = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30]
    for i in range(len(noise)):
        data_save_noisy = SaveDataNoisy(noise=noise[i])
        data_save_noisy.save_POD(sess, x_pos, f_ph, u_ph, u_pred, data, num_noisy, h)

    np.savetxt('./Output/train_loss', train_loss)
    np.savetxt('./Output/test_loss', test_loss)
    
    ## Plotting the loss history
    plt.rcParams.update({'font.size': 15})
    num_epoch = train_loss.shape[0]
    x = np.linspace(1, num_epoch, num_epoch)
    fig = plt.figure(constrained_layout=False, figsize=(6, 6))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0])
    ax.plot(x, train_loss[:,0], color='blue', label='Training Loss')
    ax.plot(x, test_loss[:,0], color='red', label='Testing Loss')
    #ax.set_yscale('log')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epochs')
    ax.legend(loc='upper left')
    plt.tight_layout()
    fig.savefig('./Output/loss_both.png')

    fig = plt.figure(constrained_layout=False, figsize=(6, 6))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0])
    ax.plot(x, train_loss[:,0], color='blue', label='Training Loss')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epochs')
    ax.legend(loc='upper left')
    plt.tight_layout()
    fig.savefig('./Output/loss_train.png')

    fig = plt.figure(constrained_layout=False, figsize=(6, 6))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0])
    ax.plot(x, test_loss[:,0], color='red', label='Testing Loss')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epochs')
    ax.legend(loc='upper left')
    plt.tight_layout()
    fig.savefig('./Output/loss_test.png')


if __name__ == "__main__":
    main()
