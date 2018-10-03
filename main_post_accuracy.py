import numpy as np
import matplotlib.pyplot as plt

test_data = np.loadtxt('./acc_test_data_original_form.dat')
test_data = 255-test_data 
test_data = (test_data>128)+0
print(test_data.shape)
   
pred_data = np.loadtxt('./acc_pred_by_vae_100_original_form.dat')
pred_data = 255-pred_data 
pred_data = (pred_data>128)+0
print(pred_data.shape)


num_pred_data = np.loadtxt('./num_acc_pred_data.dat')
num_pred_data = 255-num_pred_data 
num_pred_data = (num_pred_data>128)+0
print(num_pred_data.shape)

if True:
    # acc = 1-np.mean(np.abs(pred_data-num_pred_data))
    acc_list = []
    for i in range(len(test_data)):
        image0 = test_data[i,:].reshape([256,64])[:,32:64]
        image1 = num_pred_data[i,:].reshape([256,64])[:,32:64]
        acc_list.append(np.mean(np.abs(image0-image1)))

    acc = 1-np.mean(np.array(acc_list))
    print(acc)

plot_list = [0, 4, 8, 9, 11, 14, 15, 16, 22, 31, 103, 699, 705, 722, 858]

n_examples = len(plot_list)
plot_test_data = test_data[plot_list]
plot_pred_data = pred_data[plot_list]
plot_num_pred_data = num_pred_data[plot_list]

if False:
    fig, axs = plt.subplots(3, n_examples)
    for i in range(n_examples):
        axs[0][i].imshow(np.reshape(plot_test_data[i, :], (256, 64)))
        axs[0][i].axis('off')
        axs[1][i].imshow(np.reshape(plot_pred_data[i, :], (256, 64)))
        axs[1][i].axis('off')
        axs[2][i].imshow(np.reshape(plot_num_pred_data[i, :], (256, 64)))
        axs[2][i].axis('off')
    plt.savefig('pred.png') 
    plt.show()
    
if True:
    fig, axs = plt.subplots(4, n_examples)
    for i in range(n_examples):
        final_image = plot_test_data[i, :].reshape([256,64])[:,32:64]
        pred_init_image = plot_pred_data[i,:].reshape([256,64])[:,0:32]
        smooth_init_image = plot_num_pred_data[i,:].reshape([256,64])[:,0:32]
        num_final_image = plot_num_pred_data[i,:].reshape([256,64])[:,32:64]
        
        final_image = np.pad(final_image,((0,0),(32,0)),'symmetric')
        pred_init_image = np.pad(pred_init_image,((0,0),(32,0)),'symmetric')
        smooth_init_image = np.pad(smooth_init_image,((0,0),(32,0)),'symmetric')
        num_final_image = np.pad(num_final_image,((0,0),(32,0)),'symmetric')
        
        axs[0][i].imshow(final_image,cmap='gray')
        axs[0][i].axis('off')
        axs[1][i].imshow(pred_init_image,cmap='gray')
        axs[1][i].axis('off')
        axs[2][i].imshow(smooth_init_image,cmap='gray')
        axs[2][i].axis('off')
        axs[3][i].imshow(num_final_image,cmap='gray')
        axs[3][i].axis('off')
    plt.savefig('pred.png')
    plt.show()
    
