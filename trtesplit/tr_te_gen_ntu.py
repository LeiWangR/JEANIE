import numpy as np
import random
import h5py
import os
import csv
import pandas as pd 


# define total number of action classes
# for NTU RGBD 60  -- 60 action classes
# for NTU RGBD 120 -- 120 action classes

num_class = 60
# num_class = 120

dataset_name = 'NTU-RGBD-' + str(num_class)

if os.path.isfile('./'+ dataset_name + '_fsl_train.txt') and os.path.isfile('./'+ dataset_name + '_fsl_test.txt'):
    os.remove(dataset_name + '_fsl_train.txt')
    os.remove(dataset_name + '_fsl_test.txt')
    print('old files deleted.')

class_id = np.arange(1, num_class + 1, 1)


if num_class == 60:
    selected_id = [1, 7, 13, 19, 25, 31, 37, 43, 49, 55]
else:
    # ntu120
    selected_id = [1, 7, 13, 19, 25, 31, 37, 43, 49, 55, 61, 67, 73, 79, 85, 91, 97, 103, 109, 115]
tr_total_class = num_class - len(selected_id)

print('selected class ID: ', selected_id)

# hdf5 file path
hdf5_path = '/g/data/cp23/lw4988/'

hdf5_dataset_name = dataset_name + '.hdf5'
with h5py.File(os.path.join(hdf5_path, hdf5_dataset_name), "r") as ff:
    with open(dataset_name + '_fsl_train.txt', 'a') as tr:
        with open(dataset_name + '_fsl_test.txt', 'a') as te:
            for a_key in ff.keys():

                # for ntu60 & 120
                if int(a_key[17:20]) in selected_id:
                    te.writelines(a_key)
                    te.write('\n')
                else:
                    tr.writelines(a_key)
                    tr.write('\n')

# rank the actions
# count how many videos per class
def ranktxt(old_txt, new_txt):
    list = []
    with open(old_txt, 'r') as f:
        for line in f:
            # for ntu
            
            line = line[16:20] + line[0:16]
            list.append(line.strip())

    with open(new_txt, 'w') as f:
        for item in sorted(list):
            f.writelines(item)
            f.writelines('\n')
        f.close()



csv_list = [dataset_name + '_fsl_train.txt', dataset_name + '_fsl_test.txt']

for idx in csv_list:
    # print(idx, ' --- ')
    ranktxt(idx, idx)

print('done!')


trte_count = np.zeros((num_class, ))

for idx in csv_list:
    print(idx, ' --- ')
    with open(idx, 'r') as f:
        for line in f:
            # for ntu
            act_id = int(line[1:4])
            # print(act_id)
            # needs to -1 here for the index
            trte_count[act_id-1] = trte_count[act_id-1] + 1
trte_count = trte_count.astype(np.int)


tr_count = trte_count[0:tr_total_class]
te_count = trte_count[tr_total_class: ]

np.save(dataset_name + '_tr_count.npy', tr_count)
np.save(dataset_name + '_te_count.npy', te_count)

print('--- test ---')
tr_count = np.load(dataset_name + '_tr_count.npy')
te_count = np.load(dataset_name + '_te_count.npy')

print(tr_count, ' | sum: ', np.sum(tr_count), ' \n', te_count, ' | sum: ', np.sum(te_count))

'''
# Lei's test
# should without headers here for both files
tr_data= pd.read_csv(dataset_name + '_fsl_train.txt', header=None)
te_data= pd.read_csv(dataset_name + '_fsl_test.txt', header=None)

print(tr_data.shape, te_data.shape)

print(tr_data[-1:])
'''
