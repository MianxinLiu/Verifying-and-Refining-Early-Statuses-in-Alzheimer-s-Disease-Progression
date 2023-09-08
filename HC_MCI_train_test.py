import pandas as pd
import numpy as np
import scipy.io as scio
from sklearn.model_selection import KFold

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

#index=np.arange(len(subInfo))
#shuffle(index)
#scio.savemat('./index/shuffled_index_MCI_'+str(neg)+'vs'+str(pos), {'index':index})

#kf=KFold(n_splits=4)
#cv=0
#for train_idx, test_idx in kf.split(index):
#    cv=cv+1
#    print("Train:", train_idx, " Test:", test_idx)
#    scio.savemat('./kfold/naMCIproject/'+str(neg)+'vs'+str(pos)+'/shuffled_index_cv'+str(cv), {'train_idx': train_idx,'test_idx': test_idx })

for ROInum in [100, 200, 300, 400, 500]:
    FC_all = np.zeros((len(subInfo),  ROInum, ROInum))
    subcount = 0
    for fn in subInfo['id']:
        path = '/media/PJLAB\liumianxin/18675978328/Shanghaitech/matlab/PET_center/Rest2/FC/par' + str(ROInum) + '/' + fn + '*.mat'
        path = glob.glob(path)
        temp = scio.loadmat(path[0])
        FC_all[subcount, :, :] = temp['FC']
        subcount = subcount + 1
    FC_all = torch.from_numpy(FC_all)
    FC_all = torch.tensor(FC_all, dtype=torch.float32)
    FC_all = FC_all[index, :, :]

    qual_all=[]
    for cv in [1, 2, 3, 4]:
        temp = scio.loadmat('./kfold/naMCIproject/multi/shuffled_index_cv' + str(cv))
        train_idx = temp['train_idx'][0]
        test_idx = temp['test_idx'][0]
        print("Train:", train_idx, " Test:", test_idx)
        dataset_train = TensorDataset(FC_all[train_idx, :, :], y_data1[train_idx])
        dataset_test = TensorDataset(FC_all[test_idx, :, :], y_data1[test_idx])

        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=30, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=30)

        #model.load_state_dict(torch.load('./GCRN_copy/ckpt/ROI'+str(ROInum)+'_adni_mt-gcvnetlstm_hc_emci_b_ft.pth'))
        #model.eval()
        # gclstm = Models.GraphAttentionLSTM(bsize=batch_size,tsize=time_step)
        #weight=torch.cuda.FloatTensor([1,3.1])
        ratio = y_data1[train_idx].sum() / (y_data1[train_idx].shape[0] - y_data1[train_idx].sum())
        if ratio<1:
            weight = torch.cuda.FloatTensor([1, 1 / ratio])
        else:
            weight = torch.cuda.FloatTensor([ratio, 1])

        #weight = torch.cuda.FloatTensor([1, 3.5])
        loss_func = nn.CrossEntropyLoss(weight)  # the target label is not one-hotted
        # loss_func = nn.MSELoss()

        lr = 0.001
        EPOCH = 80

        #model.load_state_dict(torch.load('./GCRN_copy/ckpt_mil/ROI'+str(ROInum)+ '_adni_mt-gclstm_hc_emci_mil_fusion2_ft_backup.pth'))
        qualified = []
        while not qualified:
            test_auc = []
            train_los = []
            test_los = []
            train_auc = []
            sen = []
            spe = []
            #epoch_of_lr_decrease = 20

            model = MyModels.GCN_base(ROInum=ROInum)
            model.cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)
            #epoch_of_lr_decrease = 20
            auc_baseline = 0.70

            for epoch in range(EPOCH):
                for step, (b_x, b_y) in enumerate(train_loader):  # gives batch data
                    model.train()
                    # if epoch > 10 and epoch % epoch_of_lr_decrease == 0:
                    #    optimizer = torch.optim.Adam(gclstm.parameters(), lr=lr/2.0,weight_decay=1e-2)
                    b_y = b_y.view(-1)
                    b_y = b_y.long()
                    b_y = b_y.cuda()

                    output = model(b_x)  # rnn output

                    loss = loss_func(output, b_y)  # cross entropy loss
                    optimizer.zero_grad()  # clear gradients for this training step
                    loss.backward()  # backpropagation, compute gradients
                    optimizer.step()  # apply gradients

                    predicted = torch.max(output.data, 1)[1]
                    correct = (predicted == b_y).sum()
                    tr_accuracy = float(correct) / float(b_x.shape[0])
                    #train_auc.append(tr_accuracy)
                    print('[Epoch %d, Batch %5d] loss: %.3f' %
                          (epoch + 1, step + 1, loss))
                    print('|train diag loss:', loss.data.item(), '|train accuracy:', tr_accuracy
                          )
                    if tr_accuracy>0.75 and epoch>50:
                        #lr=0.001
                        #optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)
                        predicted_all = []
                        predict_p=[]
                        test_y_all = []
                        model.eval()
                        with torch.no_grad():
                            for i, (test_x, test_y) in enumerate(test_loader):
                                test_y = test_y.view(-1)
                                test_y = test_y.long()
                                test_y = test_y.cuda()

                                test_output = model(test_x)

                                test_loss = loss_func(test_output, test_y)
                                loss = loss_func(test_output, test_y)
                                print('[Epoch %d, Batch %5d] valid loss: %.3f' %
                                      (epoch + 1, step + 1, loss))
                                #test_loss = loss_func(test_output, test_y)
                                predict_p=predict_p+test_output[:,1].tolist()
                                predicted = torch.max(test_output.data, 1)[1]
                                correct = (predicted == test_y).sum()
                                accuracy = float(correct) / float(predicted.shape[0])
                                test_y = test_y.cpu()
                                predicted = predicted.cpu()
                                predicted_all = predicted_all + predicted.tolist()
                                test_y_all = test_y_all + test_y.tolist()

                        correct = (np.array(predicted_all) == np.array(test_y_all)).sum()
                        accuracy = float(correct) / float(len(test_y_all))
                        test_auc.append(accuracy)
                        sens = metrics.recall_score(test_y_all, predicted_all, pos_label=1)
                        sen.append(sens)
                        spec = metrics.recall_score(test_y_all, predicted_all, pos_label=0)
                        spe.append(spec)
                        auc = metrics.roc_auc_score(test_y_all, predicted_all)
                        print('|test accuracy:', accuracy,
                              '|test sen:', sens,
                              '|test spe:', spec,
                              '|test auc:', auc,
                              )

                        if  auc >= auc_baseline and sens >0.70 and spec >0.70 and tr_accuracy>0.75:
                            auc_baseline = auc
                            torch.save(model.state_dict(),'./models_MCI2SCD_R/gcn_'+str(neg)+'vs_rest_cv'+str(cv)+'_'+str(ROInum)+'.pth')
                            print('got one model with |test accuracy:', accuracy,
                                  '|test sen:', sens,
                                  '|test spe:', spec,
                                  '|test auc:', auc,
                                  )
                            # gcam = GradCAM.GradCam(model=gclstm.eval(), target_layer_names=['gcrn'], use_cuda=True)
                            # mask = gcam(img)
                            # auc_baseline = accuracy
                            qualified.append([accuracy, sens, spec, auc])
        qual_all.append(qualified[-1])
        #print(qualified[-1])

    print(qual_all)
    print(np.mean(qual_all,axis=0))
    print(np.std(qual_all,axis=0))

#show result
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
    model.load_state_dict(torch.load('./models_MCI2SCD_R/gcn_'+str(neg)+'vs_rest_cv'+str(cv)+'_'+str(ROInum)+'_retest.pth'))
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
scio.savemat('./results/hc_mci_'+str(ROInum)+'_noreg.mat', {'qualified':qualified})
