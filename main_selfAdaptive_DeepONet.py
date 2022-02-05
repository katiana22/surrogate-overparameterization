'''
Manuscript Associated: New developments and comparisons of manifold-based surrogates with deep neural operators
Authors: Katiana Kontolati, PhD Candidate, Johns Hopkins University
         Somdatta Goswami, Postdoctoral Researcher, Brown University
Tensorflow Version Required: TF1.15     
This should be used for the sharp data    

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
from utils.dataset import DataSet_SA as DataSet
from utils.savedata import SaveData
from utils.savedataOOD import SaveDataOOD
from utils.savedatanoisy import SaveDataNoisy

np.random.seed(1234)
p = 100 # output dimension of Branch/Trunk (latent dimension)
num = 28*28
#fnn in CNN
layer_B = [num, 128, 128, p]
#trunk net
layer_T = [3, 128, 128, 128, p]

#resolution
h = 28
w = 28

#batch_size
bs = 200

#size of input for Trunk net
nx = h
nt = 10
x_num = nt*nx*nx
beta = 0.01
def main():
    data = DataSet(nx, bs)
    x_train, f_train, u_train, Xmin, Xmax = data.minibatch()
    
    x = tf.constant(x_train, dtype=tf.float32)
    
    lambda_u0 = tf.Variable(tf.random_normal(shape = [1, 1], dtype = tf.float32), dtype = tf.float32)
    lambda_u6 = tf.Variable(tf.random_normal(shape = [1, 1], dtype = tf.float32), dtype = tf.float32)
    lambda_u7 = tf.Variable(tf.random_normal(shape = [1, 1], dtype = tf.float32), dtype = tf.float32)
    
    f_ph = tf.placeholder(shape=[None, 1, num], dtype=tf.float32)
    u_ph = tf.placeholder(shape=[None, x_num, 1], dtype=tf.float32) #[bs, x_num, 1]
    learning_rate = tf.placeholder(tf.float32, shape=[])
    
    fnn_model = FNN()
    W_B, b_B = fnn_model.hyper_initial(layer_B)
    u_B = fnn_model.fnn_B(W_B, b_B, f_ph)
    # u_B = tf.tile(u_B, [1, x_num, 1])  
    
    #Trunk net
    fnn_model = FNN()
    W_T, b_T = fnn_model.hyper_initial(layer_T)
    u_T = fnn_model.fnn(W_T, b_T, x, Xmin, Xmax)

    #inner product
    u_nn = tf.einsum('bij,nj->bnj', u_B, u_T)
    u_pred = tf.reduce_sum(u_nn, axis=-1, keepdims=True)

    regularizers = fnn_model.l2_regularizer(W_B)
    lambda_u0_t = tf.math.sin(np.pi*lambda_u0*tf.exp(lambda_u0))
    lambda_u6_t = tf.math.sin(np.pi*lambda_u6*tf.exp(lambda_u6))
    lambda_u7_t = tf.math.sin(np.pi*lambda_u7*tf.exp(lambda_u7))

    loss = tf.reduce_mean(tf.square(u_ph - u_pred)) + \
           tf.reduce_mean(tf.square(lambda_u6_t)*tf.square(u_ph[:,6*num:7*num,:] - u_pred[:,6*num:7*num,:])) + \
           tf.reduce_mean(tf.square(lambda_u7_t)*tf.square(u_ph[:,7*num:8*num,:] - u_pred[:,7*num:8*num,:])) + \
           tf.reduce_mean(tf.square(lambda_u0_t)*tf.square(u_ph[:,0:num,:] - u_pred[:,0:num,:])) + beta*regularizers
    
    optimizer1 = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1 = 0.99)
    optimizer2 = tf.train.AdamOptimizer(learning_rate=5.0e-1, beta1 = 0.99, beta2 = 0.99)
    
    grads_W_B = optimizer1.compute_gradients(loss, W_B)
    grads_b_B = optimizer1.compute_gradients(loss, b_B)
    grads_W_T = optimizer1.compute_gradients(loss, W_T)
    grads_b_T = optimizer1.compute_gradients(loss, b_T)
    grads_lambda_u0 = optimizer2.compute_gradients(loss, [lambda_u0])
    grads_lambda_u6 = optimizer2.compute_gradients(loss, [lambda_u6])
    grads_lambda_u7 = optimizer2.compute_gradients(loss, [lambda_u7])
    
    grads_lambda_u0_minus = [(-gv[0], gv[1]) for gv in grads_lambda_u0]
    grads_lambda_u6_minus = [(-gv[0], gv[1]) for gv in grads_lambda_u6]
    grads_lambda_u7_minus = [(-gv[0], gv[1]) for gv in grads_lambda_u7]

    op_W_B = optimizer1.apply_gradients(grads_W_B)
    op_b_B = optimizer1.apply_gradients(grads_b_B)
    op_W_T = optimizer1.apply_gradients(grads_W_T)
    op_b_T = optimizer1.apply_gradients(grads_b_T)
    op_lamU0 = optimizer2.apply_gradients(grads_lambda_u0_minus)
    op_lamU6 = optimizer2.apply_gradients(grads_lambda_u6_minus)
    op_lamU7 = optimizer2.apply_gradients(grads_lambda_u7_minus)
    
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
    sess.run(tf.global_variables_initializer())
    
    n = 0
    nmax = 5000  # epochs
    start_time = time.perf_counter()
    time_step_0 = time.perf_counter()
    
    train_loss = np.zeros((nmax+1, 1))
    test_loss = np.zeros((nmax+1, 1))    
    while n <= nmax:
        
        if n <30000:
            lr = 0.001
        elif (n < 60000):
            lr = 0.0005
        elif (n < 100000):
            lr = 0.0001
        else:
            lr = 0.00005
            
        x_train, f_train, u_train, _, _ = data.minibatch()
        train_dict={f_ph: f_train, u_ph: u_train, learning_rate: lr}
        sess.run([op_W_B,op_b_B,op_W_T,op_b_T], train_dict)
        sess.run([op_lamU0, op_lamU6, op_lamU7], train_dict)
               
        loss_, lambda_u0_, lambda_u6_, lambda_u7_= sess.run([loss,lambda_u0, lambda_u6, lambda_u7], feed_dict=train_dict)
        
        if n%1 == 0:
            test_id, x_test, f_test, u_test = data.testbatch(bs)
            u_test_ = sess.run(u_pred, feed_dict={f_ph: f_test})
            u_test = data.decoder(u_test)
            u_test_ = data.decoder(u_test_)
            err = np.mean(np.linalg.norm(u_test_ - u_test, 2, axis=1)/np.linalg.norm(u_test, 2, axis=1))
            time_step_1000 = time.perf_counter()
            T = time_step_1000 - time_step_0
            print('Step: %d, Loss: %.4e, Test L2 error: %.4f, Time (secs): %.4f'%(n, loss_, err, T))            
            time_step_0 = time.perf_counter()
    
        train_loss[n,0] = loss_
        test_loss[n,0] = err
        n += 1
    stop_time = time.perf_counter()
    print('Elapsed time (secs): %.3f'%(stop_time - start_time))
    
    # Save results for test data
    data_save = SaveData()
    num_test = 200
    data_save.save_SA(sess, x, fnn_model, W_T, b_T, W_B, b_B, Xmin, Xmax, f_ph, u_ph, data, num_test)
    
    # Save results for OOD data
    data_save_ood = SaveDataOOD()
    num_ood = 100
    data_save_ood.save_SA(sess, x, fnn_model, W_T, b_T, W_B, b_B, Xmin, Xmax, f_ph, u_ph, data, num_ood)

    # Save results for noisy data
    num_noisy = 200
    noise = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30]
    for i in range(len(noise)):
        data_save = SaveDataNoisy(noise=noise[i])
        data_save.save_SA(sess, x, fnn_model, W_T, b_T, W_B, b_B, Xmin, Xmax, f_ph, u_ph, data, num_noisy)
        
    ## Plotting the loss history
    num_epoch = train_loss.shape[0]
    x = np.linspace(1, num_epoch, num_epoch)
    fig = plt.figure(constrained_layout=False, figsize=(6, 6))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0])
    ax.plot(x, test_loss[:,0], color='red', label='Testing Loss')
    ax.set_yscale('log')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epochs')
    ax.legend(loc='upper left')
    fig.savefig('./Output/loss_his.png')

if __name__ == "__main__":
    main()
