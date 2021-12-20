import numpy as np
import random
import h5py
import os
import csv
import pandas as pd 


# define total number of action classes
# for 3DActionPairs, 12 classes
# for MSRAction3D, 20 classes
# for UWA3DActivity 30 classes

# num_class = 12
# num_class = 20
num_class = 30

# dataset_name = '3DActionPairs'
# dataset_name = 'MSRAction3D'
# dataset_name = 'UWA3DActivity'
dataset_name = 'UWA3DMultiviewActivityII'

# for uwa3dmultiviewactivity only
selected_id = [1, 2]
unselected_id = [3]

filename_add = str(selected_id[0]) + '-' + str(selected_id[1]) + '-' + str(unselected_id[0])

if os.path.isfile('./'+ dataset_name + '_'  + filename_add + '_fsl_train.txt') and os.path.isfile('./'+ dataset_name + '_'  + filename_add + '_fsl_test.txt'):
    os.remove(dataset_name + '_'  + filename_add + '_fsl_train.txt')
    os.remove(dataset_name + '_'  + filename_add + '_fsl_test.txt')
    print('old files deleted.')

class_id = np.arange(1, num_class + 1, 1)
# print(class_id)
# random.shuffle(subject_id)

# selected_id = list(class_id[0:round(num_class/2)])

# unselected_id = list(class_id[round(num_class/2) : ])



print('selected class ID: ', selected_id)

# hdf5 file path
hdf5_path = '/Users/wanglei/Desktop/ssgc-vt_fsl/data'

hdf5_dataset_name = dataset_name + '.hdf5'
with h5py.File(os.path.join(hdf5_path, hdf5_dataset_name), "r") as ff:
    with open(dataset_name + '_'  + filename_add + '_fsl_train.txt', 'a') as tr:
        with open(dataset_name + '_'  + filename_add + '_fsl_test.txt', 'a') as te:
            for a_key in ff.keys():

                # for 3dactionpairs, msraction3d
                # if int(a_key[1:3]) in selected_id:
                # for uwa3dactivity multiview
                if int(a_key[13:15]) in selected_id:
                # for UWA3DActivity
                # if int(a_key[4:6]) in selected_id:
                    tr.writelines(a_key)
                    tr.write('\n')
                elif int(a_key[13:15]) in unselected_id:
                    te.writelines(a_key)
                    te.write('\n')


def ranktxt(old_txt, new_txt):
    list = []
    with open(old_txt, 'r') as f:
        for line in f:
            # for uwa3dactivity
            # line = line[3:9] + line[0:3]
            # for 3dactionpairs, msraction3d we commented out above line
            list.append(line.strip())

    with open(new_txt, 'w') as f:
        for item in sorted(list):
            f.writelines(item)
            f.writelines('\n')
        f.close()



csv_list = [dataset_name + '_'  + filename_add + '_fsl_train.txt', dataset_name + '_'  + filename_add + '_fsl_test.txt']

for idx in csv_list:
    # print(idx, ' --- ')
    ranktxt(idx, idx)

print('done!')





for idx in csv_list:
    print(idx, ' --- ')
    trte_count = np.zeros((num_class, ))
    with open(idx, 'r') as f:
        for line in f:
            # for 3dactionpairs, msraction3d, uwa3dactivity
            act_id = int(line[1:3])
            # print(act_id)
            # needs to -1 here for the index
            trte_count[act_id-1] = trte_count[act_id-1] + 1
        trte_count = trte_count.astype(np.int)
    if 'train' in idx:
        print(idx)
        np.save(dataset_name + '_'  + filename_add + '_tr_count.npy', trte_count)
    else:
        print(idx, '---test')
        np.save(dataset_name + '_'  + filename_add + '_te_count.npy', trte_count)

#tr_count = trte_count[0:round(num_class/2)]
#te_count = trte_count[round(num_class/2) : ]





print('--- test ---')
tr_count = np.load(dataset_name + '_'  + filename_add + '_tr_count.npy')
te_count = np.load(dataset_name + '_'  + filename_add + '_te_count.npy')

print(tr_count, ' | sum: ', np.sum(tr_count), ' \n', te_count, ' | sum: ', np.sum(te_count))

'''
# Lei's test
# should without headers here for both files
tr_data= pd.read_csv(dataset_name + '_fsl_train.txt', header=None)
te_data= pd.read_csv(dataset_name + '_fsl_test.txt', header=None)

print(tr_data.shape, te_data.shape)

print(tr_data[-1:])
'''
