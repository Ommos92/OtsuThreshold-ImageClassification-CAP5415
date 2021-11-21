import numpy as np
import matplotlib.pyplot as plt

results_mode1 = np.loadtxt(fname='mode1_lr0.1_epoch60_batchsize10_2021_10_23-08_47_51_PM.txt')
results_mode2 = np.loadtxt(fname='mode2_lr0.1_epoch60_batchsize10_2021_10_23-09_50_11_PM.txt')
results_mode3 = np.loadtxt(fname='mode3_lr0.03_epoch60_batchsize10_2021_10_24-12_47_03_PM.txt')
results_mode4 = np.loadtxt(fname='mode4_lr0.03_epoch60_batchsize10_2021_10_24-01_28_26_PM.txt')
results_mode5 = np.loadtxt(fname='mode5_lr0.03_epoch40_batchsize10_2021_10_24-02_12_00_PM.txt')

'''
#Plot Training Loss
plt.plot(results_mode1[:,0], label='model #1')
#due to scaling issues leave out poor performing model 2
plt.plot(results_mode2[:,0], label='model #2')
plt.plot(results_mode3[:,0], label='model #3')
plt.plot(results_mode4[:,0], label='model #4')
plt.plot(results_mode5[:,0], label='model #5')

plt.title('Training Loss for MNIST ConvNet')
plt.ylabel('Training Average Loss')
plt.xlabel('Epoch number')

plt.legend()
plt.show()
'''
'''
#Plot Training Accuracy
plt.plot(results_mode1[:,1], label='model #1')
#due to scaling issues leave out poor performing model 2
plt.plot(results_mode2[:,1], label='model #2')
plt.plot(results_mode3[:,1], label='model #3')
plt.plot(results_mode4[:,1], label='model #4')
plt.plot(results_mode5[:,1], label='model #5')

plt.title('Training Accuracy for MNIST ConvNet')
plt.ylabel('Training Accuracy (%)')
plt.xlabel('Epoch number')

plt.legend()
plt.show()
'''

'''
#Plot Test Loss
plt.plot(results_mode1[:,2], label='model #1')
#due to scaling issues leave out poor performing model 2
#plt.plot(results_mode2[:,2], label='model #2')
plt.plot(results_mode3[:,2], label='model #3')
plt.plot(results_mode4[:,2], label='model #4')
plt.plot(results_mode5[:,2], label='model #5')

plt.title('Average Test Loss for MNIST ConvNet')
plt.ylabel('Average Test Loss')
plt.xlabel('Epoch number')

plt.legend()
plt.show()
'''
'''
#Plot Test Accuracy
plt.plot(results_mode1[:,3], label='model #1')
#due to scaling issues leave out poor performing model 2
#plt.plot(results_mode2[:,3], label='model #2')
plt.plot(results_mode3[:,3], label='model #3')
plt.plot(results_mode4[:,3], label='model #4')
plt.plot(results_mode5[:,3], label='model #5')

plt.title('Test Accuracy for MNIST ConvNet')
plt.ylabel('Test Accuracy (%)')
plt.xlabel('Epoch number')

plt.legend()
plt.show()
'''

max_training_accuracy = np.max(results_mode5[:,1])
max_test_accuracy = np.max(results_mode5[:,3])
print('Best Training Accuracy: %f' % max_training_accuracy)
print('Best Test Accuracy: %f' % max_test_accuracy)