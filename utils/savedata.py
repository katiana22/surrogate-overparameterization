'''
Manuscript Associated: On the influence of over-parameterization in manifold based surrogates and deep neural operators
Authors: Katiana Kontolati, PhD Candidate, Johns Hopkins University
         Somdatta Goswami, Postdoctoral Researcher, Brown University
Tensorflow Version Required: TF1.15         
'''

import tensorflow.compat.v1 as tf
import numpy as np
from utils.plotting import *
import scipy.io as io
import os

dump_test = './Output/Plots/'
os.makedirs(dump_test, exist_ok=True)

    
class SaveData:
    def __init__(self):
        pass

    def save(self, sess, x_pos, fnn_model, W, b, Xmin, Xmax, u_B, f_ph, u_ph, data, num_test):
        test_id, x_test, f_test, u_test = data.testbatch(num_test)
        x = tf.tile(x_pos[None, :, :], [num_test, 1, 1])
        u_T = fnn_model.fnn(W, b, x, Xmin, Xmax)
        test_dict = {f_ph: f_test, u_ph: u_test}
        u_nn = u_B*u_T

        u_pred = tf.reduce_sum(u_nn, axis=-1, keepdims=True)
        u_pred_ = sess.run(u_pred, feed_dict=test_dict)
        u_test = data.decoder(u_test)
        u_pred_ = data.decoder(u_pred_)
        f_test = np.reshape(f_test, (f_test.shape[0], -1))
        u_pred_ = np.reshape(u_pred_, (u_test.shape[0], u_test.shape[1]))

        U_ref = np.reshape(u_test, (u_test.shape[0], u_test.shape[1]))
        np.savetxt('./Output/u_ref', U_ref, fmt='%e')
        np.savetxt('./Output/test_id', test_id, fmt='%e')
        np.savetxt('./Output/f_test', f_test, fmt='%e')
        np.savetxt('./Output/u_pred', u_pred_, fmt='%e')

        err = np.mean(np.linalg.norm(u_pred_ - U_ref, 2, axis=1)/np.linalg.norm(U_ref, 2, axis=1))
        print('Relative L2 Error: %.5f'%(err))
        err = np.reshape(err, (-1, 1))
        np.savetxt('./Output/err_test', err, fmt='%e')

        num_rows = 28
        num_cols = 28
        nt = 20
        for i in range(0, num_test, 50):

            k_print = f_test[i,:]
            k_print = k_print.reshape(num_rows, num_cols) 
            
            disp_pred = u_pred_[i,:].reshape(nt, num_rows, num_cols)                
            disp_true = U_ref[i,:].reshape(nt, num_rows, num_cols)   

            print(f"Plotting results for test sample: {i}", flush=True)
            dataSegment = "Test"
            plotField(k_print, disp_pred, disp_true, i, dump_test, dataSegment, nt) 

    def save_SA(self, sess, x_pos, fnn_model, W_T, b_T, W_B, b_B, Xmin, Xmax, f_ph, u_ph, data, num_test):
        
        test_id, x_test, f_test, u_test = data.testbatch(num_test)
        x = tf.tile(x_pos[None, :, :], [num_test, 1, 1])
        u_T = fnn_model.fnn(W_T, b_T, x, Xmin, Xmax)
        test_dict = {f_ph: f_test, u_ph: u_test}
        u_B = fnn_model.fnn_B(W_B, b_B, f_ph)
        u_nn = u_B*u_T

        u_pred = tf.reduce_sum(u_nn, axis=-1, keepdims=True)
        u_pred_ = sess.run(u_pred, feed_dict=test_dict)
        u_test = data.decoder(u_test)
        u_pred_ = data.decoder(u_pred_)
        f_test = np.reshape(f_test, (f_test.shape[0], -1))
        u_pred_ = np.reshape(u_pred_, (u_test.shape[0], u_test.shape[1]))

        U_ref = np.reshape(u_test, (u_test.shape[0], u_test.shape[1]))
        np.savetxt('./Output/u_ref', U_ref, fmt='%e')
        np.savetxt('./Output/test_id', test_id, fmt='%e')
        np.savetxt('./Output/f_test', f_test, fmt='%e')
        np.savetxt('./Output/u_pred', u_pred_, fmt='%e')

        err = np.mean(np.linalg.norm(u_pred_ - U_ref, 2, axis=1)/np.linalg.norm(U_ref, 2, axis=1))
        print('Relative L2 Error: %.5f'%(err))
        err = np.reshape(err, (-1, 1))
        np.savetxt('./Output/err_test', err, fmt='%e')
        
        # io.savemat('Brusselator_test_DeepONet.mat', 
        #              mdict={'x_test': f_test,
        #                     'y_test': U_ref, 
        #                     'y_pred': u_pred_})
        
        num_rows = 28
        num_cols = 28
        nt = 10
        for i in range(0, num_test, 50):

            k_print = f_test[i,:]
            k_print = k_print.reshape(num_rows, num_cols) 
            
            disp_pred = u_pred_[i,:].reshape(nt, num_rows, num_cols)                
            disp_true = U_ref[i,:].reshape(nt, num_rows, num_cols)   

            print(f"Plotting results for test sample: {i}", flush=True)
            dataSegment = "Test"
            plotField(k_print, disp_pred, disp_true, i, dump_test, dataSegment, nt) 
            
    def save_POD(self, sess, x_pos, f_ph, u_ph, u_pred, data, num_test, h):
        
        test_id, x_test, f_test, u_test = data.testbatch(num_test)
        test_dict = {f_ph: f_test, u_ph: u_test}

        u_pred_ = sess.run(u_pred, feed_dict=test_dict)
        u_test = data.decoder(u_test)
        u_pred_ = data.decoder(u_pred_)
        f_test = np.reshape(f_test, (f_test.shape[0], -1))
        u_pred_ = np.reshape(u_pred_, (u_test.shape[0], u_test.shape[1]))
        U_ref = np.reshape(u_test, (u_test.shape[0], u_test.shape[1]))

        err = np.mean(np.linalg.norm(u_pred_ - U_ref, 2, axis=1)/np.linalg.norm(U_ref, 2, axis=1))
        print('Relative L2 Error: %.3f'%(err))
        err = np.reshape(err, (-1, 1))
        np.savetxt('./Output/err_test', err, fmt='%e')

        save_dict = {'u_pred': u_pred_, 'u_ref': U_ref, 'f_test': f_test, 'test_id': test_id}
        io.savemat('./Output/pred.mat', save_dict)
        
        # scipy.io.savemat('Brusselator_test_DeepONet.mat', 
        #              mdict={'x_test': f_test,
        #                     'y_test': u_test, 
        #                     'y_pred': u_pred_})

        np.savetxt('./Output/u_ref', U_ref, fmt='%e')
        np.savetxt('./Output/f_test', f_test, fmt='%e')
        np.savetxt('./Output/u_pred', u_pred_, fmt='%e')

        num_rows = 28
        num_cols = 28
        nt = 10
        for i in range(0, num_test, 50):

            k_print = f_test[i,:]
            k_print = k_print.reshape(num_rows, num_cols) 
            
            disp_pred = u_pred_[i,:].reshape(nt, num_rows, num_cols)                
            disp_true = U_ref[i,:].reshape(nt, num_rows, num_cols)   

            print(f"Plotting results for test sample: {i}", flush=True)
            dataSegment = "Test"
            plotField(k_print, disp_pred, disp_true, i, dump_test, dataSegment, nt)             
