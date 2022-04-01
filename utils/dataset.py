'''
Manuscript Associated: On the influence of over-parameterization in manifold based surrogates and deep neural operators
Authors: Katiana Kontolati, PhD Candidate, Johns Hopkins University
         Somdatta Goswami, Postdoctoral Researcher, Brown University
Tensorflow Version Required: TF1.15         
'''

import numpy as np
import scipy.io as io
np.random.seed(1234)

file1 = np.load('Load Training Data and Testing')# Training Data
file2 = np.load('Load Out_of_Distribution Data') # OOD Data

class DataSet:
    def __init__(self, num, bs):
        self.num = num
        self.bs = bs
        self.F_train, self.U_train, self.F_test, self.U_test, \
        self.X, self.u_mean, self.u_std, self.F_ood, self.U_ood, self.u_ood_mean, self.u_ood_std = self.load_data()

    def decoder(self, x):
        x = x*(self.u_std + 1.0e-9) + self.u_mean
        return x
    
    def decoder_ood(self, x):
        x = x*(self.u_ood_std + 1.0e-9) + self.u_ood_mean
        return x

    def load_data(self):
            
        # Training Data
        nt, nx, ny = 20, file1['nx'], file1['ny']
        n_samples = file1['n_samples']
        inputs = file1['inputs'].reshape(n_samples, nx, ny)
        outputs = np.array((file1['outputs'])).reshape(n_samples, nt, nx, ny)
        
        # Out of distribution data        
        n_samples_ood = file2['n_samples']
        inputs_ood = file2['inputs'].reshape(n_samples_ood, nx, ny)
        outputs_ood = np.array((file2['outputs'])).reshape(n_samples_ood, nt, nx, ny)
       
        num_train = 800
        num_test = 200
        num_ood = 200

        s, t = 28, 20
        
        f_train = inputs[:num_train, :, :]
        u_train = outputs[:num_train, :, :, :]   

        f_test = inputs[num_train:num_train+num_test, :, :]
        u_test = outputs[num_train:num_train+num_test, :, :, :] 
        
        f_ood = inputs_ood[:num_ood]
        u_ood = outputs_ood[:num_ood]
        
        x = np.linspace(0, 1, s)
        y = np.linspace(0, 1, s)
        z = np.linspace(0, 1, t)

        zz, xx, yy = np.meshgrid(z, x, y, indexing='ij')

        xx = np.reshape(xx, (-1, 1)) 
        yy = np.reshape(yy, (-1, 1)) 
        zz = np.reshape(zz, (-1, 1)) 

        X = np.hstack((zz, xx, yy)) # shape=[t*s*s,3]
                                            
        # compute mean values
        f_train_mean = np.mean(f_train, 0)
        f_train_std = np.std(f_train, 0)
        u_train_mean = np.mean(u_train, 0)
        u_train_std = np.std(u_train, 0)
        
        # OOD data
        f_ood_mean = np.mean(f_ood, 0)
        f_ood_std = np.std(f_ood, 0)
        u_ood_mean = np.mean(u_ood, 0)
        u_ood_std = np.std(u_ood, 0)
        
        num_res = t*s*s 
        
        # Reshape
        f_train_mean = np.reshape(f_train_mean, (-1, s, s, 1))
        f_train_std = np.reshape(f_train_std, (-1, s, s, 1))
        u_train_mean = np.reshape(u_train_mean, (-1, num_res, 1))
        u_train_std = np.reshape(u_train_std, (-1, num_res, 1))
        
        # OOD data
        f_ood_mean = np.reshape(f_ood_mean, (-1, s, s, 1))
        f_ood_std = np.reshape(f_ood_std, (-1, s, s, 1))
        u_ood_mean = np.reshape(u_ood_mean, (-1, num_res, 1))
        u_ood_std = np.reshape(u_ood_std, (-1, num_res, 1))
        
        #  Mean normalization of train data
        F_train = np.reshape(f_train, (-1, s, s, 1))
        F_train = (F_train - f_train_mean)/(f_train_std + 1.0e-9)
        U_train = np.reshape(u_train, (-1, num_res, 1))
        U_train = (U_train - u_train_mean)/(u_train_std + 1.0e-9)

        #  Mean normalization of test data (using the mean and std of train)
        F_test = np.reshape(f_test, (-1, s, s, 1))
        F_test = (F_test - f_train_mean)/(f_train_std + 1.0e-9)
        U_test = np.reshape(u_test, (-1, num_res, 1))
        U_test = (U_test - u_train_mean)/(u_train_std + 1.0e-9)
        
        #  Mean normalization of ood data 
        F_ood = np.reshape(f_ood, (-1, s, s, 1))
        F_ood = (F_ood - f_ood_mean)/(f_ood_std + 1.0e-9)
        U_ood = np.reshape(u_ood, (-1, num_res, 1))
        U_ood = (U_ood - u_ood_mean)/(u_ood_std + 1.0e-9)

        return F_train, U_train, F_test, U_test, X, u_train_mean, u_train_std, F_ood, U_ood, u_ood_mean, u_ood_std

        
    def minibatch(self):

        batch_id = np.random.choice(self.F_train.shape[0], self.bs, replace=False)

        f_train = self.F_train[batch_id]
        u_train = self.U_train[batch_id]
        x_train = self.X

        Xmin = np.array([ 0., 0., 0.]).reshape((-1, 3))
        Xmax = np.array([ 1., 1., 1.]).reshape((-1, 3))

        return x_train, f_train, u_train, Xmin, Xmax

    def testbatch(self, num_test):
        
        batch_id = np.random.choice(self.F_test.shape[0], num_test, replace=False)
        f_test = self.F_test[batch_id]
        u_test = self.U_test[batch_id]

        x_test = self.X

        batch_id = np.reshape(batch_id, (-1, 1))

        return batch_id, x_test, f_test, u_test
    
    def oodbatch(self, num_ood):
        batch_id = np.random.choice(self.U_ood.shape[0], num_ood, replace=False)
        f_ood = self.F_ood[batch_id]
        u_ood = self.U_ood[batch_id]

        x_ood = self.X

        batch_id = np.reshape(batch_id, (-1, 1))

        return batch_id, x_ood, f_ood, u_ood
   
    def noisybatch(self, num_noisy, noise):
        batch_id = np.random.choice(self.F_test.shape[0], num_noisy, replace=False)
        f_noisy_ = self.F_test[batch_id]
        u_noisy = self.U_test[batch_id]

        nx,ny = f_noisy_.shape[1], f_noisy_.shape[2]
        f_noisy = np.zeros((num_noisy, nx, ny, 1))
        for k in range(num_noisy):
            for i in range(nx):
                for j in range(ny):
                    f_noisy[k, i, j, :] = f_noisy_[k, i, j, :] + np.random.normal(0, np.abs(noise*f_noisy_[k, i, j, :]))
         
        x_noisy = self.X

        batch_id = np.reshape(batch_id, (-1, 1))

        return batch_id, x_noisy, f_noisy, u_noisy
 
class DataSet_SA:
    def __init__(self, num, bs):
        self.num = num
        self.bs = bs
        self.F_train, self.U_train, self.F_test, self.U_test, \
        self.X, self.u_mean, self.u_std, self.F_ood, self.U_ood, self.u_ood_mean, self.u_ood_std = self.load_data()

    def decoder(self, x):
        x = x*(self.u_std + 1.0e-9) + self.u_mean
        return x
    
    def decoder_ood(self, x):
        x = x*(self.u_ood_std + 1.0e-9) + self.u_ood_mean
        return x

    def load_data(self):
    

        
        nt, nx, ny = 10, file1['nx'], file1['ny']
        n_samples = file1['n_samples']
        inputs = file1['inputs'].reshape(n_samples, nx, ny)
        outputs = np.array((file1['outputs'])).reshape(n_samples, nt, nx, ny)
        
        n_samples_ood = file2['n_samples']
        inputs_ood = file2['inputs'].reshape(n_samples_ood, nx, ny)
        outputs_ood = np.array((file2['outputs'])).reshape(n_samples_ood, nt, nx, ny)
       
        num_train = 800
        num_test = 200
        num_ood = 200

        s, t = 28, 10
        
        f_train = inputs[:num_train, :, :]
        u_train = outputs[:num_train, :, :, :]   

        f_test = inputs[num_train:num_train+num_test, :, :]
        u_test = outputs[num_train:num_train+num_test, :, :, :] 
        
        f_ood = inputs_ood[:num_ood]
        u_ood = outputs_ood[:num_ood]
        
        x = np.linspace(0, 1, s)
        y = np.linspace(0, 1, s)
        z = np.linspace(0, 1, t)

        zz, xx, yy = np.meshgrid(z, x, y, indexing='ij')

        xx = np.reshape(xx, (-1, 1)) # flatten
        yy = np.reshape(yy, (-1, 1)) # flatten
        zz = np.reshape(zz, (-1, 1)) # flatten

        X = np.hstack((zz, xx, yy)) # shape=[t*s*s,3]
                                            
        # compute mean values
        f_train_mean = np.mean(f_train, 0)
        f_train_std = np.std(f_train, 0)
        u_train_mean = np.mean(u_train, 0)
        u_train_std = np.std(u_train, 0)
        
        # OOD data
        f_ood_mean = np.mean(f_ood, 0)
        f_ood_std = np.std(f_ood, 0)
        u_ood_mean = np.mean(u_ood, 0)
        u_ood_std = np.std(u_ood, 0)
        
        num_res = t*s*s # total output dimension

        
        # Reshape
        f_train_mean = np.reshape(f_train_mean, (-1, 1, s*s))
        f_train_std = np.reshape(f_train_std, (-1, 1, s*s))
        u_train_mean = np.reshape(u_train_mean, (-1, num_res, 1))
        u_train_std = np.reshape(u_train_std, (-1, num_res, 1))
        
        # OOD data
        f_ood_mean = np.reshape(f_ood_mean, (-1, 1, s*s))
        f_ood_std = np.reshape(f_ood_std, (-1, 1, s*s))
        u_ood_mean = np.reshape(u_ood_mean, (-1, num_res, 1))
        u_ood_std = np.reshape(u_ood_std, (-1, num_res, 1))
        
        #  Mean normalization of train data
        F_train = np.reshape(f_train, (-1, 1, s*s))
        F_train = (F_train - f_train_mean)/(f_train_std + 1.0e-9)
        U_train = np.reshape(u_train, (-1, num_res, 1))
        U_train = (U_train - u_train_mean)/(u_train_std + 1.0e-9)

        #  Mean normalization of test data (using the mean and std of train)
        F_test = np.reshape(f_test, (-1, 1, s*s))
        F_test = (F_test - f_train_mean)/(f_train_std + 1.0e-9)
        U_test = np.reshape(u_test, (-1, num_res, 1))
        U_test = (U_test - u_train_mean)/(u_train_std + 1.0e-9)
        
        #  Mean normalization of ood data 
        F_ood = np.reshape(f_ood, (-1, 1, s*s))
        F_ood = (F_ood - f_ood_mean)/(f_ood_std + 1.0e-9)
        U_ood = np.reshape(u_ood, (-1, num_res, 1))
        U_ood = (U_ood - u_ood_mean)/(u_ood_std + 1.0e-9)

        return F_train, U_train, F_test, U_test, X, u_train_mean, u_train_std, F_ood, U_ood, u_ood_mean, u_ood_std

        
    def minibatch(self):

        batch_id = np.random.choice(self.F_train.shape[0], self.bs, replace=False)

        f_train = self.F_train[batch_id]
        u_train = self.U_train[batch_id]
        x_train = self.X

        Xmin = np.array([ 0., 0., 0.]).reshape((-1, 3))
        Xmax = np.array([ 1., 1., 1.]).reshape((-1, 3))

        return x_train, f_train, u_train, Xmin, Xmax

    def testbatch(self, num_test):
        batch_id = np.random.choice(self.F_test.shape[0], num_test, replace=False)
        f_test = self.F_test[batch_id]
        u_test = self.U_test[batch_id]
        x_test = self.X

        batch_id = np.reshape(batch_id, (-1, 1))

        return batch_id, x_test, f_test, u_test
    
    def oodbatch(self, num_ood):
        batch_id = np.random.choice(self.U_ood.shape[0], num_ood, replace=False)
        f_ood = self.F_ood[batch_id]
        u_ood = self.U_ood[batch_id]

        x_ood = self.X

        batch_id = np.reshape(batch_id, (-1, 1))

        return batch_id, x_ood, f_ood, u_ood
   
    def noisybatch(self, num_noisy, noise):
        batch_id = np.random.choice(self.F_test.shape[0], num_noisy, replace=False)
        f_noisy_ = self.F_test[batch_id]
        u_noisy = self.U_test[batch_id]

        nx,ny = f_noisy_.shape[1], f_noisy_.shape[2]
        f_noisy = np.zeros((num_noisy, 1, nx*ny))
        for k in range(num_noisy):
            for i in range(nx*ny):
                f_noisy[k, :, i] = f_noisy_[k, :, i] + np.random.normal(0, np.abs(noise*f_noisy_[k, :, i]))
        
        x_noisy = self.X

        batch_id = np.reshape(batch_id, (-1, 1))

        return batch_id, x_noisy, f_noisy, u_noisy
    
class DataSet_POD:
    
    def __init__(self, num, bs, modes):
        self.num = num
        self.bs = bs
        self.modes = modes
        self.F_train, self.U_train, self.F_test, self.U_test, \
        self.X, self.u_mean, self.u_std, \
        self.u_basis, self.lam_u, self.F_ood, self.U_ood, self.u_ood_mean, self.u_ood_std  = self.load_data()

    def PODbasis(self):
        s = 28*28*10
        print(self.u_basis.shape)

        u_basis_out = np.reshape(self.u_basis.T, (-1, s, 1))
        u_basis_out = self.decoder(u_basis_out)
        u_basis_out = u_basis_out - self.u_mean
        u_basis_out = np.reshape(u_basis_out, (-1, s))
        save_dict = {'u_basis': u_basis_out, 'lam_u': self.lam_u}
        io.savemat('./Output/basis.mat', save_dict)
        return self.u_basis

    def decoder(self, x):
        x = x*(self.u_std + 1.0e-9) + self.u_mean
        return x

    def decoder_ood(self, x):
        x = x*(self.u_ood_std + 1.0e-9) + self.u_ood_mean
        return x

    def load_data(self):
        
        nt, nx, ny = 10, file1['nx'], file1['ny']
        n_samples = file1['n_samples']
        inputs = file1['inputs'].reshape(n_samples, nx, ny)
        outputs = np.array((file1['outputs'])).reshape(n_samples, nt, nx, ny)
        
        n_samples_ood = file2['n_samples']
        inputs_ood = file2['inputs'].reshape(n_samples_ood, nx, ny)
        outputs_ood = np.array((file2['outputs'])).reshape(n_samples_ood, nt, nx, ny)

        num_train = 800
        num_test = 200
        num_ood = 200

        s, t = 28, 10
        
        f_train = inputs[:num_train, :, :]
        u_train = outputs[:num_train, :, :, :]    
       
        f_test = inputs[num_train:num_train+num_test, :, :]
        u_test = outputs[num_train:num_train+num_test, :, :, :]

        f_ood = inputs_ood[:num_ood]
        u_ood = outputs_ood[:num_ood]

        x = np.linspace(0, 1, s)
        y = np.linspace(0, 1, s)
        z = np.linspace(0, 1, t)

        zz, xx, yy = np.meshgrid(z, x, y, indexing='ij')

        xx = np.reshape(xx, (-1, 1)) # flatten
        yy = np.reshape(yy, (-1, 1)) # flatten
        zz = np.reshape(zz, (-1, 1)) # flatten

        X = np.hstack((zz, xx, yy)) # shape=[t*s*s,3]

        f_train_mean = np.mean(f_train, 0)
        f_train_std = np.std(f_train, 0)
        u_train_mean = np.mean(u_train, 0)
        u_train_std = np.std(u_train, 0)

        # OOD data
        f_ood_mean = np.mean(f_ood, 0)
        f_ood_std = np.std(f_ood, 0)
        u_ood_mean = np.mean(u_ood, 0)
        u_ood_std = np.std(u_ood, 0)

        num_res = t*s*s # total output dimension

        # Train data
        f_train_mean = np.reshape(f_train_mean, (-1, s, s, 1))
        f_train_std = np.reshape(f_train_std, (-1, s, s, 1))
        u_train_mean = np.reshape(u_train_mean, (-1, s*s*t, 1))
        u_train_std = np.reshape(u_train_std, (-1, s*s*t, 1))

        # OOD data
        f_ood_mean = np.reshape(f_ood_mean, (-1, s, s, 1))
        f_ood_std = np.reshape(f_ood_std, (-1, s, s, 1))
        u_ood_mean = np.reshape(u_ood_mean, (-1, num_res, 1))
        u_ood_std = np.reshape(u_ood_std, (-1, num_res, 1))

        #  Mean normalization of train data 
        F_train = np.reshape(f_train, (-1, s, s, 1))
        F_train = (F_train - f_train_mean)/(f_train_std + 1.0e-9)
        U_train = np.reshape(u_train, (-1, num_res, 1))
        U_train = (U_train - u_train_mean)/(u_train_std + 1.0e-9)

        #  Mean normalization of test data 
        F_test = np.reshape(f_test, (-1, s, s, 1))
        F_test = (F_test - f_train_mean)/(f_train_std + 1.0e-9)
        U_test = np.reshape(u_test, (-1, num_res, 1))
        U_test = (U_test - u_train_mean)/(u_train_std + 1.0e-9)
        
        #  Mean normalization of ood data 
        F_ood = np.reshape(f_ood, (-1, s, s, 1))
        F_ood = (F_ood - f_ood_mean)/(f_ood_std + 1.0e-9)
        U_ood = np.reshape(u_ood, (-1, num_res, 1))
        U_ood = (U_ood - u_ood_mean)/(u_ood_std + 1.0e-9)

        # Train data
        U = np.reshape(U_train, (-1, num_res))
        C_u = 1./(num_train-1)*np.matmul(U.T, U)
        lam_u, phi_u = np.linalg.eigh(C_u)

        lam_u = np.flip(lam_u)
        phi_u = np.fliplr(phi_u)
        phi_u = phi_u*np.sqrt(num_res)

        u_cumsum = np.cumsum(lam_u)
        u_per = u_cumsum[self.modes-1]/u_cumsum[-1]

        u_basis = phi_u[:, :self.modes]        

        return F_train, U_train, F_test, U_test, X, u_train_mean, u_train_std, u_basis, lam_u, F_ood, U_ood, u_ood_mean, u_ood_std
        
    def minibatch(self):
        
        batch_id = np.random.choice(self.F_train.shape[0], self.bs, replace=False)
        f_train = [self.F_train[i:i+1] for i in batch_id]
        f_train = np.concatenate(f_train, axis=0)
        u_train = [self.U_train[i:i+1] for i in batch_id]
        u_train = np.concatenate(u_train, axis=0)

        x_train = self.X

        Xmin = np.array([ 0.,  0.]).reshape((-1, 2))
        Xmax = np.array([ 1.,  1.]).reshape((-1, 2))


        return x_train, f_train, u_train, Xmin, Xmax

    def testbatch(self, num_test):

        batch_id = np.arange(num_test)
        f_test = [self.F_test[i:i+1] for i in batch_id]
        f_test = np.concatenate(f_test, axis=0)
        u_test = [self.U_test[i:i+1] for i in batch_id]
        u_test = np.concatenate(u_test, axis=0)

        x_test = self.X

        batch_id = np.reshape(batch_id, (-1, 1))

        return batch_id, x_test, f_test, u_test


    def oodbatch(self, num_ood):

        batch_id = np.arange(num_ood)
        f_ood = [self.F_ood[i:i+1] for i in batch_id]
        f_ood = np.concatenate(f_ood, axis=0)
        u_ood = [self.U_ood[i:i+1] for i in batch_id]
        u_ood = np.concatenate(u_ood, axis=0)

        batch_id = np.reshape(batch_id, (-1, 1))
        x_ood = self.X
        return batch_id, x_ood, f_ood, u_ood


    def noisybatch(self, num_noisy, noise):
        batch_id = np.random.choice(self.F_test.shape[0], num_noisy, replace=False)
        f_noisy_ = [self.F_test[i:i+1] for i in batch_id]
        f_noisy_ = np.concatenate(f_noisy_, axis=0)
        u_noisy = [self.U_test[i:i+1] for i in batch_id]
        u_noisy = np.concatenate(u_noisy, axis=0)

        nx,ny = f_noisy_.shape[1], f_noisy_.shape[2]
        f_noisy = np.zeros((num_noisy, nx, ny, 1))
        for k in range(num_noisy):
            for i in range(nx):
                for j in range(ny):
                    f_noisy[k, i, j, :] = f_noisy_[k, i, j, :] + np.random.normal(0, np.abs(noise*f_noisy_[k, i, j, :]))

        x_noisy = self.X

        batch_id = np.reshape(batch_id, (-1, 1))

        return batch_id, x_noisy, f_noisy, u_noisy    
