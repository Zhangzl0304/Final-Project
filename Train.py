import pandas as pd
from torch import nn
import numpy as np
import sklearn
from sklearn.metrics import classification_report, auc, roc_curve
from sklearn.model_selection import train_test_split
import torch
from dataset import data_label_2, DatasetFromFolder
import matplotlib.pyplot as plt
from Model import CNN
from torch.utils.tensorboard import SummaryWriter


train_data = DatasetFromFolder('CNN')
data_train = []
for i in range(1488):
    data_train.append(train_data[i])
data_train = np.array(data_train)



writer = SummaryWriter(log_dir='loss')

max_epoch = 200
max_iter = 1



data_input = data_train
data_input = (data_input-data_input.mean(axis=0))/data_input.std(axis=0)
label_input = data_label_2


roc_list = []
prc_list = []
precision1_list = []
recall1_list = []
acc_list = []


# SA = Self_Attention(250)
# data_input = torch.from_numpy(data_train)
# data_input = data_input.permute(0, 2, 1)
# out = SA(data_input.to(torch.float32))
# print(out, out.size())


CNN = CNN()

Model = CNN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Model.to(device)



for ite in range(max_iter):


    train_idx, test_idx, labels_model, labels_test = train_test_split(range(data_input.shape[0]), label_input,
                                                                      stratify=label_input)
    print(len(train_idx), len(test_idx))
    print('iteration {}'.format(ite+1))

    data_train = data_input[train_idx, :]
    label_train = labels_model
    data_test = data_input[test_idx, :]

    data_train = torch.from_numpy(data_train)
    label_train = np.atleast_2d(label_train).T
    label_train = torch.from_numpy(np.array(label_train))



    data_test = torch.from_numpy(data_test)
    labels_test = torch.from_numpy(np.array(labels_test))


    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(Model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-6, verbose=True)

    # best_nn_model.apply(weight_reset)
    Model.train()
    for epoch in range(max_epoch):
        print('epoch:', epoch)
        pred = Model(data_train.to(torch.float32))
        loss = criterion(pred, label_train.to(torch.float32))
        writer.add_scalar("Loss", scalar_value=loss,
                          global_step=epoch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step(loss)
    writer.close()
    Model.eval()
    test_pred = Model(data_test.to(torch.float32))
    test_pred = test_pred.cpu()
    test_pred_np = test_pred.detach().numpy()


    precision, recall, threshold = sklearn.metrics.precision_recall_curve(labels_test, test_pred_np)
    prc = auc(recall, precision)
    prc_list.append(prc)
    roc = sklearn.metrics.roc_auc_score(labels_test, test_pred_np)
    roc_list.append(roc)
    predict01 = []
    for i in range(len(test_pred_np)):
       if test_pred_np[i] >= 0.5:
           predict01.append(1)
       else:
           predict01.append(0)
    report = classification_report(labels_test, predict01, output_dict=True)
    precision1 = report['1']['precision']
    precision1_list.append(precision1)
    recall1 = report['1']['recall']
    recall1_list.append(recall1)
    acc = report['accuracy']
    acc_list.append(acc)
    fpr, tpr, threshold = roc_curve(labels_test, test_pred_np)
    roc_auc = auc(fpr, tpr)
    # plt.figure()
    # lw = 2
    # plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr,
             label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1],   linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])


print(prc_list, 'PRC', np.mean(prc_list), np.std(prc_list))
print(roc_list, 'ROC', np.mean(roc_list), np.std(roc_list))
print(precision1_list, 'precision', np.mean(precision1_list), np.std(precision1_list))
print(recall1_list, 'recall', np.mean(recall1_list), np.std(recall1_list))
print(acc_list, 'acc', np.mean(acc_list), np.std(acc_list))

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC of CNN')
plt.show()