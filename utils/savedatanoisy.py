'''
Manuscript Associated: New developments and comparisons of manifold-based surrogates with deep neural operators
Authors: Katiana Kontolati, PhD Candidate, Johns Hopkins University
         Somdatta Goswami, Postdoctoral Researcher, Brown University
Tensorflow Version Required: TF1.15         
'''

import tensorflow.compat.v1 as tf
import numpy as np
from utils.plotting import *
import os

dump_test = './Output/Plots/Noise/'
os.makedirs(dump_test, exist_ok=True)
    
class SaveDataNoisy:
    def __init__(self, noise):
        self.noise = noise

    def save(self, sess, x_pos, fnn_model, W, b, Xmin, Xmax, u_B, f_ph, u_ph, data, num_noisy):
        test_id, x_noisy, f_noisy, u_noisy = data.noisybatch(num_noisy, self.noise)
        x = tf.tile(x_pos[None, :, :], [num_noisy, 1, 1])
        u_T = fnn_model.fnn(W, b, x, Xmin, Xmax)
        test_dict = {f_ph: f_noisy, u_ph: u_noisy}
        u_nn = u_B*u_T

        u_pred = tf.reduce_sum(u_nn, axis=-1, keepdims=True)
        u_pred_ = sess.run(u_pred, feed_dict=test_dict)
        u_noisy = data.decoder(u_noisy)
        u_pred_ = data.decoder(u_pred_)
        f_noisy = np.reshape(f_noisy, (f_noisy.shape[0], -1))
        u_pred_ = np.reshape(u_pred_, (u_noisy.shape[0], u_noisy.shape[1]))

        U_ref = np.reshape(u_noisy, (u_noisy.shape[0], u_noisy.shape[1]))
        np.savetxt('./Output/u_ref_{}'.format(self.noise), U_ref, fmt='%e')
        np.savetxt('./Output/test_id_{}'.format(self.noise), test_id, fmt='%e')
        np.savetxt('./Output/f_test_{}'.format(self.noise), f_noisy, fmt='%e')
        np.savetxt('./Output/u_pred_{}'.format(self.noise), u_pred_, fmt='%e')

        err = np.mean(np.linalg.norm(u_pred_ - U_ref, 2, axis=1)/np.linalg.norm(U_ref, 2, axis=1))
        print('Relative L2 Error (Noise): %.5f'%(err))
        err = np.reshape(err, (-1, 1))
        np.savetxt('./Output/err_noise_{}'.format(self.noise), err, fmt='%e')
        
        num_rows = 28
        num_cols = 28
        nt = 20
        for i in range(0, num_noisy, 50):

            k_print = f_noisy[i,:]
            k_print = k_print.reshape(num_rows, num_cols) 
            
            disp_pred = u_pred_[i,:].reshape(nt, num_rows, num_cols)                
            disp_true = U_ref[i,:].reshape(nt, num_rows, num_cols)   

            print(f"Plotting results for noisy sample: {i}", flush=True)
            dataSegment = "Noise_{}".format(self.noise)
            plotField(k_print, disp_pred, disp_true, i, dump_test, dataSegment, nt) 
            
    def save_SA(self, sess, x_pos, fnn_model, W_T, b_T, W_B, b_B, Xmin, Xmax, f_ph, u_ph, data, num_noisy):
        
        test_id, x_noisy, f_noisy, u_noisy = data.noisybatch(num_noisy, self.noise)
        x = tf.tile(x_pos[None, :, :], [num_noisy, 1, 1])
        u_T = fnn_model.fnn(W_T, b_T, x, Xmin, Xmax)
        test_dict = {f_ph: f_noisy, u_ph: u_noisy}
        u_B = fnn_model.fnn_B(W_B, b_B, f_ph)
        u_nn = u_B*u_T

        u_pred = tf.reduce_sum(u_nn, axis=-1, keepdims=True)
        u_pred_ = sess.run(u_pred, feed_dict=test_dict)
        u_noisy = data.decoder(u_noisy)
        u_pred_ = data.decoder(u_pred_)
        f_noisy = np.reshape(f_noisy, (f_noisy.shape[0], -1))
        u_pred_ = np.reshape(u_pred_, (u_noisy.shape[0], u_noisy.shape[1]))

        U_ref = np.reshape(u_noisy, (u_noisy.shape[0], u_noisy.shape[1]))
        np.savetxt('./Output/u_ref_{}'.format(self.noise), U_ref, fmt='%e')
        np.savetxt('./Output/test_id_{}'.format(self.noise), test_id, fmt='%e')
        np.savetxt('./Output/f_test_{}'.format(self.noise), f_noisy, fmt='%e')
        np.savetxt('./Output/u_pred_{}'.format(self.noise), u_pred_, fmt='%e')

        err = np.mean(np.linalg.norm(u_pred_ - U_ref, 2, axis=1)/np.linalg.norm(U_ref, 2, axis=1))
        print('Relative L2 Error (Noise): %.5f'%(err))
        err = np.reshape(err, (-1, 1))
        np.savetxt('./Output/err_noise_{}'.format(self.noise), err, fmt='%e')

        num_rows = 28
        num_cols = 28
        nt = 10
        for i in range(0, num_noisy, 50):

            k_print = f_noisy[i,:]
            k_print = k_print.reshape(num_rows, num_cols)

            disp_pred = u_pred_[i,:].reshape(nt, num_rows, num_cols)
            disp_true = U_ref[i,:].reshape(nt, num_rows, num_cols)

            print(f"Plotting results for noisy sample: {i}", flush=True)
            dataSegment = "Noise_{}".format(self.noise)
            plotField(k_print, disp_pred, disp_true, i, dump_test, dataSegment, nt)       

    def save_POD(self, sess, x_pos, f_ph, u_ph, u_pred, data, num_noisy, h):
        noisy_id, x_noisy, f_noisy, u_noisy = data.noisybatch(num_noisy, self.noise)

        test_dict = {f_ph: f_noisy, u_ph: u_noisy}
        '''
        xs = 2*(x - Xmin)/(Xmax - Xmin) - 1
        u_pred = (xs[:, :, 0:1] - 1)*(xs[:, :, 0:1] + 1)*(xs[:, :, 1:2] - 1)*(xs[:, :, 1:2] + 1)*u_pred
        '''
        u_pred_ = sess.run(u_pred, feed_dict=test_dict)
        u_noisy = data.decoder(u_noisy)
        u_noisy_ = data.decoder(u_pred_)
        f_noisy = np.reshape(f_noisy, (f_noisy.shape[0], -1))
        u_noisy_ = np.reshape(u_noisy_, (u_noisy.shape[0], u_noisy.shape[1]))
        U_ref = np.reshape(u_noisy, (u_noisy.shape[0], u_noisy.shape[1]))

        np.savetxt('./Output/u_ref_{}'.format(self.noise), U_ref, fmt='%e')
        np.savetxt('./Output/f_test_{}'.format(self.noise), f_noisy, fmt='%e')
        np.savetxt('./Output/u_pred_{}'.format(self.noise), u_noisy_, fmt='%e')

        err = np.mean(np.linalg.norm(u_noisy_ - U_ref, 2, axis=1)/np.linalg.norm(U_ref, 2, axis=1))
        print('Relative L2 Error (Noise): %.5f'%(err))
        err = np.reshape(err, (-1, 1))
        np.savetxt('./Output/err_noise_{}'.format(self.noise), err, fmt='%e')
