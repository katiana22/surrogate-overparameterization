import numpy as np
from matplotlib import pylab as plt

train_loss = np.loadtxt('./Output/train_loss')
test_loss = np.loadtxt('./Output/test_loss')

inputs_test = np.loadtxt('./Output/f_test')
pred = np.loadtxt('./Output/u_pred')
true = np.loadtxt('./Output/u_ref')
print(inputs_test.shape, pred.shape, true.shape)

num_test = inputs_test.shape[0]

plt.rcParams.update({'font.size': 18})
fig = plt.figure(figsize=(7, 6))
epochs = train_loss.shape[0]
x = np.linspace(1, epochs, epochs)
plt.plot(x, train_loss, label='train loss')
#plt.yscale('log')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('Output/train_loss.png', dpi=250)

fig = plt.figure(figsize=(7, 6))
plt.plot(x, test_loss, label='test loss')
#plt.yscale('log')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('Output/test_loss.png', dpi=250)

nx, ny = 28, 28
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
xx, yy = np.meshgrid(x, y)

index = 10
tot = int(nx*ny)

snapshots = 5
time = np.linspace(0,1,snapshots)
print(time)
get = np.linspace(0, pred.shape[1], snapshots+1)
get = [int(x) for x in get]

plt.rcParams.update({'font.size': 12.5})
th = 0.06
fig, axs = plt.subplots(3, snapshots, figsize=(14.5,6), constrained_layout=True)
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 1
for col in range(snapshots):
    for row in range(3):
        ax = axs[row, col]
        if row == 0:
            ss1 = true[index, get[col]:get[col]+tot]
            pcm = ax.contourf(xx, yy, ss1.reshape(nx,ny), levels=np.linspace(np.min(ss1)-th, np.max(ss1)+th, 100), cmap='jet')
            cbar = plt.colorbar(pcm, ax=ax, format='%.1f', ticks=np.linspace(np.min(ss1)-th, np.max(ss1)+th , 5))
            ax.set_title(r'$t={}$'.format(time[col]))
        if row == 1:
            ss2 = pred[index, get[col]:get[col]+tot]
            pcm = ax.contourf(xx, yy, ss2.reshape(nx,ny), levels=np.linspace(np.min(ss1)-th, np.max(ss1)+th, 100), cmap='jet')
            cbar = plt.colorbar(pcm, ax=ax, format='%.1f', ticks=np.linspace(np.min(ss1)-th, np.max(ss1)+th , 5))
        if row == 2:
            errors = np.abs((pred - true)/true)
            ss = errors[index, get[col]:get[col]+tot]
            pcm = ax.contourf(xx, yy, ss.reshape(nx,ny), levels=100, cmap='jet')
            plt.colorbar(pcm, ax=ax, format='%.0e', ticks=np.linspace(np.min(ss), np.max(ss) , 5))

        if row == 2:
            ax.set_xlabel(r'$x$', fontsize=13)
        if col ==0 and row ==0:
            ax.set_ylabel('Reference \n y', fontsize=13)
        if col == 0 and row==1:
            ax.set_ylabel('DeepONet \n y', fontsize=13)
        if col == 0 and row==2:
            ax.set_ylabel('Error \n y', fontsize=13)
        ax.locator_params(axis='y', nbins=3)
        ax.locator_params(axis='x', nbins=3)
plt.savefig('Output/comparison.png', bbox_inches='tight', dpi=500)
