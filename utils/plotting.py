'''
Manuscript Associated: New developments and comparisons of manifold-based surrogates with deep neural operators
Authors: Katiana Kontolati, PhD Candidate, Johns Hopkins University
         Somdatta Goswami, Postdoctoral Researcher, Brown University
Tensorflow Version Required: TF1.15         
'''

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
matplotlib.use('Agg')

def plotField(k_print, disp_pred, disp_true, istep, folder, segment, nt):

        time_step_ = [i for i in range(nt)]
        tt = [i/nt for i in range(1, nt+1)]

        for k in range(len(time_step_)):
            
            time_step = time_step_[k]

            fig = plt.figure(figsize=(14,3))
            plt.rcParams.update({'font.size': 13.5})
            #fig.suptitle(segment+'ing case: ' + str(istep+1), fontsize=16)
            
            x = np.linspace(0, 1, k_print.shape[0])
            y = np.linspace(0, 1, k_print.shape[0])
            xx, yy = np.meshgrid(x, y)
            
            gs = fig.add_gridspec(1,4)
            plt.subplots_adjust(left = 0.1, bottom = 0.1, right = 0.9, top = 0.5, wspace = 0.4, hspace = 0.1)
            
            ax = fig.add_subplot(gs[0,0])        
            h = ax.contourf(xx, yy, k_print, levels=100, cmap='rainbow')
            ax.set_title('v(t=0,x)')
            ax.set_ylabel('Test case: ' + str(istep+1))
            divider = make_axes_locatable(ax)
            ax.locator_params(axis='y', nbins=3)
            ax.locator_params(axis='x', nbins=3)
            cax = divider.append_axes("right", size="10%", pad=0.1)
            plt.colorbar(h, ax=ax, cax=cax, format='%.2f', ticks=np.linspace(np.min(k_print), np.amax(k_print), 5)) 
            
            ax = fig.add_subplot(gs[0,1])        
            h = ax.contourf(xx, yy, disp_pred[time_step,:,:], levels=np.linspace(np.min(disp_pred[time_step,:,:]), np.amax(disp_pred[time_step,:,:]), 100), cmap='rainbow')
            ax.set_title('Pred v(t={},x)'.format(tt[k]))
            divider = make_axes_locatable(ax)
            ax.locator_params(axis='y', nbins=3)
            ax.locator_params(axis='x', nbins=3)
            cax = divider.append_axes("right", size="10%", pad=0.1)
            plt.colorbar(h, ax=ax, cax=cax, format='%.2f', ticks=np.linspace(np.min(disp_pred[time_step,:,:]), np.amax(disp_pred[time_step,:,:]), 5))
            
            ax = fig.add_subplot(gs[0,2])
            h = ax.contourf(xx, yy, disp_true[time_step,:,:], levels=np.linspace(np.min(disp_true[time_step,:,:]), np.amax(disp_true[time_step,:,:]), 100), cmap='rainbow')
            ax.set_title('True v(t={},x)'.format(tt[k]))
            divider = make_axes_locatable(ax)
            ax.locator_params(axis='y', nbins=3)
            ax.locator_params(axis='x', nbins=3)
            cax = divider.append_axes("right", size="10%", pad=0.1)
            plt.colorbar(h, ax=ax, cax=cax, format='%.2f', ticks=np.linspace(np.min(disp_true[time_step,:,:]), np.amax(disp_true[time_step,:,:]), 5))
            
            ax = fig.add_subplot(gs[0,3])
            error = abs((disp_pred[time_step,:,:] - disp_true[time_step,:,:])/disp_true[time_step,:,:])
            h = ax.contourf(xx, yy, error, levels=100, cmap='rainbow')
            ax.set_title('Error')
            divider = make_axes_locatable(ax)
            ax.locator_params(axis='y', nbins=3)
            ax.locator_params(axis='x', nbins=3)
            cax = divider.append_axes("right", size="10%", pad=0.1)
            plt.colorbar(h, ax=ax, cax=cax, format='%.2f', ticks=np.linspace(np.min(error), np.amax(error), 5))
            
            fig.tight_layout()
            fig.savefig(folder + '/' + segment + '_sample_' + str(istep) + '_time_step_' + str(time_step+1) + '.png')
            plt.close()
        
