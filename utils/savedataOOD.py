'''
Manuscript Associated: On the influence of over-parameterization in manifold based surrogates and deep neural operators
Authors: Katiana Kontolati, PhD Candidate, Johns Hopkins University
         Somdatta Goswami, Postdoctoral Researcher, Brown University
Tensorflow Version Required: TF1.15         
'''

import tensorflow.compat.v1 as tf
import numpy as np
from utils.plotting import *
import os

dump_test = './Output/Plots/OOD/'
os.makedirs(dump_test, exist_ok=True)

    
class SaveDataOOD:
    def __init__(self):
        pass

    def save(self, sess, x_pos, fnn_model, W, b, Xmin, Xmax, u_B, f_ph, u_ph, data, num_ood):
        test_id, x_ood, f_ood, u_ood = data.oodbatch(num_ood)
        x = tf.tile(x_pos[None, :, :], [num_ood, 1, 1])
        u_T = fnn_model.fnn(W, b, x, Xmin, Xmax)
        test_dict = {f_ph: f_ood, u_ph: u_ood}
        u_nn = u_B*u_T

        u_pred = tf.reduce_sum(u_nn, axis=-1, keepdims=True)
        u_pred_ = sess.run(u_pred, feed_dict=test_dict)
        u_ood = data.decoder_ood(u_ood)
        u_pred_ = data.decoder_ood(u_pred_)
        f_ood = np.reshape(f_ood, (f_ood.shape[0], -1))
        u_pred_ = np.reshape(u_pred_, (u_ood.shape[0], u_ood.shape[1]))

        U_ref = np.reshape(u_ood, (u_ood.shape[0], u_ood.shape[1]))
        np.savetxt('./Output/u_ref_ood', U_ref, fmt='%e')
        np.savetxt('./Output/test_id_ood', test_id, fmt='%e')
        np.savetxt('./Output/f_test_ood', f_ood, fmt='%e')
        np.savetxt('./Output/u_pred_ood', u_pred_, fmt='%e')

        err = np.mean(np.linalg.norm(u_pred_ - U_ref, 2, axis=1)/np.linalg.norm(U_ref, 2, axis=1))
        print('Relative L2 Error (OOD): %.5f'%(err))
        err = np.reshape(err, (-1, 1))
        np.savetxt('./Output/err_OOD', err, fmt='%e')

        num_rows = 28
        num_cols = 28
        nt = 20
        for i in range(0, num_ood, 40):

            k_print = f_ood[i,:]
            k_print = k_print.reshape(num_rows, num_cols) 
            
            disp_pred = u_pred_[i,:].reshape(nt, num_rows, num_cols)                
            disp_true = U_ref[i,:].reshape(nt, num_rows, num_cols)   

            print(f"Plotting results for ood sample: {i}", flush=True)
            dataSegment = "OOD"
            plotField(k_print, disp_pred, disp_true, i, dump_test, dataSegment, nt) 
            
    def save_SA(self, sess, x_pos, fnn_model, W_T, b_T, W_B, b_B, Xmin, Xmax, f_ph, u_ph, data, num_test):

        test_id, x_test, f_test, u_test = data.oodbatch(num_test)
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
        print('Relative L2 Error for OOD data: %.5f'%(err))
        err = np.reshape(err, (-1, 1))
        np.savetxt('./Output/err_ood', err, fmt='%e')
        
        num_rows = 28
        num_cols = 28
        nt = 10
        for i in range(0, num_test, 50):

            k_print = f_test[i,:]
            k_print = k_print.reshape(num_rows, num_cols) 
            
            disp_pred = u_pred_[i,:].reshape(nt, num_rows, num_cols)                
            disp_true = U_ref[i,:].reshape(nt, num_rows, num_cols)   

            print(f"Plotting results for test sample: {i}", flush=True)
            dataSegment = "OOD"
            plotField(k_print, disp_pred, disp_true, i, dump_test, dataSegment, nt)          

    def save_POD(self, sess, x_pos, f_ph, u_ph, u_pred, data, num_ood, h):
        
        ood_id, x_ood, f_ood, u_ood = data.oodbatch(num_ood)

        test_dict = {f_ph: f_ood, u_ph: u_ood}
        u_pred_ = sess.run(u_pred, feed_dict=test_dict)
        u_ood = data.decoder(u_ood)
        u_ood_ = data.decoder(u_pred_)
        f_ood = np.reshape(f_ood, (f_ood.shape[0], -1))
        u_ood_ = np.reshape(u_ood_, (u_ood.shape[0], u_ood.shape[1]))
        U_ref = np.reshape(u_ood, (u_ood.shape[0], u_ood.shape[1]))

        err = np.mean(np.linalg.norm(u_ood_ - U_ref, 2, axis=1)/np.linalg.norm(U_ref, 2, axis=1))
        print('Relative L2 Error for OOD data: %.5f'%(err))
        err = np.reshape(err, (-1, 1))
        np.savetxt('./Output/err_ood', err, fmt='%e')
