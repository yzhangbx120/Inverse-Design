![Image text](https://raw.githubusercontent.com/yzhangbx120/Inverse-Design/master/pred.png)


# Inverse-Design

The data can be download from "https://drive.google.com/open?id=1Xh7ghdEDwLi36jUpIYPQTfD8wG5QSaef".
Unzip data.zip and copy all the files in data folder to the main folder.

# Meaning of data files

checkpoint and model* are data saved data files used for tensorflow

data.dat: training data

acc_test_data_original_form.dat: testing data

acc_pred_by_vae_100_original_form.dat: inverse design result for testing data

num_acc_pred_data.dat: the final shapes after surface diffusion with the prediced initial shapes for testing data

Inverse design: python main.py

Calculate accuracy and design results: python main_post_accuracy.py
