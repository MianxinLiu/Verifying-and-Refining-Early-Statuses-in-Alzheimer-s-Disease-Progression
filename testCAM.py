import pandas as pd
import numpy as np
import scipy.io as scio

import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset
import MyModels
from sklearn import metrics
from random import shuffle
import GradCAM
import glob

neg=0

subInfo = pd.read_csv('./SCD-MCI-subinfo_revise_final_2.csv')
subInfo = subInfo[(subInfo['grouping']==0) | (subInfo['grouping']==1) | (subInfo['grouping']==2)]

y_data1 = torch.from_numpy(subInfo['grouping'].to_numpy())
y_data1 = torch.tensor(y_data1, dtype=torch.float32)

y_data1[y_data1 != neg] = 9
y_data1[y_data1 == neg] = 0
y_data1[y_data1 == 9] = 1

temp = scio.loadmat('./index/shuffled_index_multiMCI')
index = temp['index'][0]
y_data1 = y_data1[index]


ROInum=500
FC_all = np.zeros((len(subInfo),  ROInum, ROInum))
subcount = 0
for fn in subInfo['id']:
    path = '/media/PJLAB\liumianxin/18675978328/Shanghaitech/matlab/PET_center/Rest2/FC_reg4/par' + str(ROInum) + '/' + fn + '*.mat'
    path = glob.glob(path)
    temp = scio.loadmat(path[0])
    FC_all[subcount, :, :] = temp['FC']
    subcount = subcount + 1
FC_all = torch.from_numpy(FC_all)
FC_all = torch.tensor(FC_all, dtype=torch.float32)
FC_all = FC_all[index, :, :]


qualified=[]
for cv in [1,2,3,4]:
    temp = scio.loadmat('./kfold/naMCIproject/multi/shuffled_index_cv' + str(cv))
    test_idx = temp['test_idx'][0]
    dataset_test = TensorDataset(FC_all[test_idx, :, :], y_data1[test_idx])
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=30)

    model = MyModels.GCN_base(ROInum=ROInum)
    model.load_state_dict(torch.load('./models_MCI2SCD/gcn_'+str(neg)+'vs_rest_cv'+str(cv)+'_'+str(ROInum)+'.pth'))
    model.cuda()
    predicted_all = []
    test_y_all = []
    model.eval()
    with torch.no_grad():
        for i, (test_x, test_y) in enumerate(test_loader):
            test_y = test_y.view(-1)
            test_y = test_y.long()
            test_y = test_y.cuda()

            test_output = model(test_x)
            #test_loss = loss_func(test_output, test_y)
            predicted = torch.max(test_output.data, 1)[1]
            correct = (predicted == test_y).sum()
            accuracy = float(correct) / float(predicted.shape[0])
            test_y = test_y.cpu()
            predicted = predicted.cpu()
            predicted_all = predicted_all + predicted.tolist()
            test_y_all = test_y_all + test_y.tolist()

    correct = (np.array(predicted_all) == np.array(test_y_all)).sum()
    accuracy = float(correct) / float(len(test_y_all))
    #test_auc.append(accuracy)
    sens = metrics.recall_score(test_y_all, predicted_all, pos_label=1)
    #sen.append(sens)
    spec = metrics.recall_score(test_y_all, predicted_all, pos_label=0)
    #spe.append(spec)
    auc = metrics.roc_auc_score(test_y_all, predicted_all)
    print('|test accuracy:', accuracy,
          '|test sen:', sens,
          '|test spe:', spec,
          '|test auc:', auc,
          )
    qualified.append([accuracy, sens, spec, auc])

print(np.mean(qualified, axis=0))
print(np.std(qualified, axis=0))
filename_save = './gradcam/Metrics_hc_mci_'+str(ROInum)+'_noreg.mat'
scio.savemat(filename_save,{'qualified':qualified})



mask_all=np.zeros((len(subInfo),ROInum,4))
for cv in ([1,2,3,4]):
    model = MyModels.GCN_base(ROInum = ROInum)
    model.load_state_dict(
        torch.load('./models_MCI2SCD/gcn_' + str(neg) + 'vs' + str(pos) + '_cv' + str(cv) + '_' + str(ROInum) + '.pth'))
    model.cuda()
    target_index = 1
    gcam = GradCAM.GradCam(model=model.eval(), ROInum= ROInum, target_layer_names=['gcn'], use_cuda=True)
    FC_all.requires_grad=True
    for i in range(len(subInfo)):
        input=torch.zeros((1,ROInum,ROInum))
        input[0,:,:]=FC_all[i,:,:]
        mask_all[i,:,cv-1] = gcam(input, target_index)

filename_save = './gradcam/CAMmask_amci_namci_'+str(ROInum)+'.mat'
scio.savemat(filename_save,{'mask_all':mask_all})

# to check GradCam in correctly predicted sub only
filename_save = './gradcam/group_amci_namci_noreg.mat'
y_data1=y_data1.tolist()
y_data1=np.array(y_data1)
scio.savemat(filename_save,{'y_data1':y_data1})



