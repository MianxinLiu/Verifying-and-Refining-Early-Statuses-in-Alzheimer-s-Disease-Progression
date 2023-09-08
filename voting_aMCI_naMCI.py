import pandas as pd
import numpy as np
import scipy.io as scio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset
import MyModels
from sklearn import metrics
from random import shuffle
import glob
import collections

neg=1
pos=2

subInfo = pd.read_csv('./SCD-MCI-subinfo_revise_final_2.csv')
subInfo = subInfo[(subInfo['grouping']==neg) | (subInfo['grouping']==pos)]

y_data1 = torch.from_numpy(subInfo['grouping'].to_numpy())
y_data1 = torch.tensor(y_data1, dtype=torch.float32)

y_data1[y_data1 != neg] = 9
y_data1[y_data1 == neg] = 0
y_data1[y_data1 == 9] = 1

temp = scio.loadmat('./index/shuffled_index_MCI_'+str(neg)+'vs'+str(pos))
index = temp['index'][0]
y_data1 = y_data1[index]

ROI =500
qualified=[]
for cv in [1,2,3,4]:
    temp = scio.loadmat('./kfold/naMCIproject/'+str(neg)+'vs'+str(pos)+'/shuffled_index_cv' + str(cv))
    test_idx = temp['test_idx'][0]
    vote=np.zeros([len(test_idx),int(ROI/100)])
    for ROInum in range(100,ROI+100,100):

        FC_all = np.zeros((len(subInfo), ROInum, ROInum))
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

        dataset_test = TensorDataset(FC_all[test_idx, :, :], y_data1[test_idx])
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=30)

        model = MyModels.GCN_base(ROInum=ROInum)
        model.load_state_dict(torch.load(
            './models_MCI2SCD/gcn_' + str(neg) + 'vs' + str(pos) + '_cv' + str(cv) + '_' + str(
                ROInum) + '.pth'))
        model.cuda()
        predicted_all = []
        predict_p = []
        test_y_all = []
        model.eval()
        with torch.no_grad():

            for i, (test_x, test_y) in enumerate(test_loader):
                test_y = test_y.view(-1)
                test_y = test_y.long()
                test_y = test_y.cuda()
                test_output= model(test_x)

                predict_p = predict_p + test_output[:, 1].tolist()
                predicted = torch.max(test_output.data, 1)[1]

                test_y = test_y.cpu()
                predicted = predicted.cpu()
                predicted_all = predicted_all + predicted.tolist()
                test_y_all = test_y_all + test_y.tolist()


        #print('|test accuracy:', accuracy,
        #      '|test sen:', sens,
        #      '|test spe:', spec,
        #      '|test auc:', auc,
        #      )
        vote[:,int((ROInum/100))-1]=np.array(predicted_all)

    countvote = np.sum(vote[:, :], axis=1)
    countvote[countvote < int(ROI / 100 / 2 + 0.5)] = 0
    countvote[countvote >= int(ROI / 100 / 2 + 0.5)] = 1

    predicted_all = countvote.tolist()
    correct = (np.array(predicted_all) == np.array(test_y_all)).sum()
    accuracy = float(correct) / float(len(test_y_all))
    sens = metrics.recall_score(test_y_all, predicted_all, pos_label=1)
    spec = metrics.recall_score(test_y_all, predicted_all, pos_label=0)
    auc = metrics.roc_auc_score(test_y_all, predicted_all)
    print('|test accuracy:', accuracy,
          '|test sen:', sens,
          '|test spe:', spec,
          '|test auc:', auc,
          )
    qualified.append([accuracy, sens, spec, auc])

print(np.mean(qualified, axis=0))
print(np.std(qualified, axis=0))
performance = qualified
scio.savemat('./results/amci_namci_multi.mat', {'qualified':qualified})


# deep feature comparison (on SCD and AD)

neg=1
pos=2

subInfo = pd.read_csv('./SCD-MCI-subinfo_revise_final_2.csv')
subInfo = subInfo[(subInfo['grouping']==3)] #3=SCD 4=AD

y_data1 = torch.from_numpy(subInfo['grouping'].to_numpy())
y_data1 = torch.tensor(y_data1, dtype=torch.float32)

ROI =500

qualified=[]
vote_fold=np.zeros([len(subInfo),4])
for cv in [1,2,3,4]:
    vote=np.zeros([len(subInfo),int(ROI/100)])

    for ROInum in range(100,ROI+100,100):
        FC_all = np.zeros((len(subInfo), ROInum, ROInum))
        subcount = 0
        for fn in subInfo['id']:
            path = '/media/PJLAB\liumianxin/18675978328/Shanghaitech/matlab/PET_center/Rest2/FC_reg4/par' + str(ROInum) + '/' + fn + '*.mat'
            path = glob.glob(path)
            temp = scio.loadmat(path[0])
            FC_all[subcount, :, :] = temp['FC']
            subcount = subcount + 1
        FC_all = torch.from_numpy(FC_all)
        FC_all = torch.tensor(FC_all, dtype=torch.float32)

        dataset_test = TensorDataset(FC_all, y_data1)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=30)

        model = MyModels.GCN_base(ROInum=ROInum)
        model.load_state_dict(torch.load(
            './models_MCI2SCD/gcn_' + str(neg) + 'vs' + str(pos) + '_cv' + str(cv) + '_' + str(
                ROInum) + '.pth'))
        model.cuda()
        predicted_all = []
        predict_p = []
        test_y_all = []
        model.eval()
        with torch.no_grad():
            for i, (test_x, test_y) in enumerate(test_loader):
                test_y = test_y.view(-1)
                test_y = test_y.long()
                test_y = test_y.cuda()

                test_output = model(test_x)
                predicted = torch.max(test_output.data, 1)[1]
                predict_p = predict_p + test_output[:, 1].tolist()
                correct = (predicted == test_y).sum()
                accuracy = float(correct) / float(predicted.shape[0])
                test_y = test_y.cpu()
                predicted = predicted.cpu()
                predicted_all = predicted_all + predicted.tolist()
                test_y_all = test_y_all + test_y.tolist()


        vote[:,int((ROInum/100))-1]=np.array(predicted_all)

    countvote = np.sum(vote[:, :], axis=1)
    countvote[countvote < int(ROI / 100 / 2 + 0.5)] = 0
    countvote[countvote >= int(ROI / 100 / 2 + 0.5)] = 1

    vote_fold[:, cv-1]=countvote.tolist()

countvote = torch.zeros((len(subInfo),2))
for ii in range(len(subInfo)):
    for cv in range(4):
        countvote[ii,int(vote_fold[ii,cv])] += performance[cv][(int(vote_fold[ii,cv])+1)]

predicted_all=torch.max(countvote.data, 1)[1]
predicted_all=predicted_all.tolist()
aMCI = (np.array(predicted_all) == 0).sum()/ float(len(predicted_all))
naMCI = (np.array(predicted_all) == 1).sum()/ float(len(predicted_all))

print('|aMCI:', aMCI,
      '|naMCI:', naMCI
      )
      
scio.savemat('./results/amci_namci_'+str(ROInum)+'_subtying_SCD.mat', {'predicted_all':predicted_all})

# three classification results are generated by integrating predictions for SCD from two models 
pred1 = scio.loadmat('./results/amci_namci_'+str(ROInum)+'_subtying_SCD.mat')
pred1 = pred1['predicted_all'][0]
pred1 = pred1 + 1
pred2 = scio.loadmat('./results/hc_mci_'+str(ROInum)+'_subtying_SCD.mat')
pred2 = pred2['predicted_all'][0]

predicted_all = pred2

for i in range(len(predicted_all)):
    if pred2[i]==1 and pred1[i]==1:
        predicted_all[i] = 1
    if pred2[i]==1 and pred1[i]==2:
        predicted_all[i] = 2

scio.savemat('./results/hc_amci_namci_'+str(ROInum)+'_subtying_SCD.mat', {'predicted_all':predicted_all})
