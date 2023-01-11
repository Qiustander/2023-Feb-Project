import matplotlib.pyplot as plt
import os.path as pth
import numpy as np
from glob import glob

folder_name = 'CSDIS4/Mujoco'
file_path = pth.join('./result', folder_name)

original_file_list = sorted(glob(file_path + '/*original*', recursive=True))
mask_file_list = sorted(glob(file_path + '/*mask*', recursive=True))
impute_file_list = sorted(glob(file_path + '/*imputation*', recursive=True))

mask = np.load(mask_file_list[0])
original_data = np.load(original_file_list[0])
gen_series = np.load(impute_file_list[0])

gen_median = np.quantile(gen_series, 0.5, axis=1)
gen_median = gen_median*(1-mask)+ mask*original_data

for channel in range(original_data.shape[-1]):
    plot_orin = original_data[0, ..., channel]
    plot_gen = gen_median[0, ..., channel]
    plt.figure(figsize = (18,9))
    plt.plot(range(plot_gen.shape[0]),plot_gen,color='b',label='Imputation')
    plt.plot(range(plot_orin.shape[0]),plot_orin,color='orange',label='True')
    #plt.xticks(range(0,df.shape[0],50),df['Date'].loc[::50],rotation=45)
    plt.xlabel('Time Point')
    plt.legend(fontsize=18)
    plt.show()