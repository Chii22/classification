import torch
from torch import nn,optim 
import torch.nn as nn

import tqdm 
import math
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from torch.utils.data import TensorDataset,DataLoader
from torchvision import transforms,models
from torchvision.utils import save_image 
from torchvision.datasets import ImageFolder 

import os
import glob
from PIL import Image
from google.colab import drive,files

from sklearn.metrics import precision_recall_fscore_support,log_loss,confusion_matrix,roc_curve, precision_recall_curve, auc,confusion_matrix, accuracy_score, precision_score,recall_score,f1_score


#ImageFolder関数を使用してDatasetを作成する
train_imgs = ImageFolder(
    "/content/drive/MyDrive/taco_and_burrito/train",
    transform = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.ToTensor()]))
test_imgs = ImageFolder(
    "/content/drive/MyDrive/taco_and_burrito/test",
    transform=transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor()]))


#DataLoaderを作成
train_loader=DataLoader(
    train_imgs,batch_size=32,shuffle=True)
test_loader=DataLoader(
    test_imgs,batch_size=32,shuffle=False)


#事前学習済みのresnet18をロード
net = models.resnet18(pretrained = True)


#全てのパラメータを微分対象外にする
for p in net.parameters():
    p.requires_grad = False
    #最後の線形層を付け替える
    fc_input_dim = net.fc.in_features
    net.fc = nn.Linear(fc_input_dim,2)


#分類学習を評価するためのloglossクラスを用意する
def logloss(true_label, predicted,cm,eps=1e-15):
    tl=true_label
    pl=predicted
    ylog=0
    p0=cm[0,0]/(cm[0,0]+cm[0,1])
    p1=cm[1,1]/(cm[1,0]+cm[1,1])
    # print("それぞれの確率は",p0,p1)
    for i in range(len(tl)):
      # true labelが1の時
        if tl[i] == 1:
            q=-math.log(p1)
      # true labelが0の時
        else:
            q=-math.log(1-p0)
        ylog=ylog+q
    ylog=ylog/len(tl) 
    # print("ylogは",ylog)
    return ylog


# 評価のためのクラスを用意する
def eval_net(net,data_loader,loss_fn,device="cpu"):
  #DropoutやBatchNormを無効化
    net.eval()
    ys=[]
    ypreds=[]
    hys=[]
    test_losses=[]
    for x,y in data_loader:
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            _, y_pred = net(x).max(1)
        #hyはnetを使って予測されたy
        hy=net(x)
        ys.append(y)
        ypreds.append(y_pred)
        hys.append(hy)
    #ミニバッチごとの予測結果などを1つにまとめる
    ys = torch.cat(ys)
    ypreds = torch.cat(ypreds)
    hys = torch.cat(hys)
    #testのloss
    test_loss=loss_fn(hys,ys).float()
    
    #今後の作業のためys,ypredsをnumpyに変換
    ysn = ys.to('cpu').detach().numpy().copy()
    ypredsn = ypreds.to('cpu').detach().numpy().copy()
    #予測精度(MSE)を計算
    acc=(ys==ypreds).float().sum()/len(ys) 
    #混同行列を作成
    cm=confusion_matrix(ysn, ypredsn)
    print("testの混合行列は\n",cm)
    # loglossクラスを使って計算
    ylogs=logloss(ysn,ypredsn,cm)
    # print('log loss = ',ylogs)
    # 適合率、再現率、F値の出力
    print('accuracy = ',accuracy_score(ysn, ypredsn))
    print('precision = ',precision_score(ysn, ypredsn))
    print('recall = ',recall_score(ysn, ypredsn))
    print('f1 score = ',f1_score(ysn, ypredsn))
    
    #ROC曲線のため行列を縦1列に変換
    ysnn=ysn.reshape(-1,1)
    ypredsnn=ypredsn.reshape(-1,1)
    # ROC曲線、PR曲線の準備
    fpr, tpr, thresholds = roc_curve(ysnn,ypredsnn)
    fpr_tpr_thresholds_df = pd.DataFrame([fpr,tpr,thresholds])
    fpr_tpr_thresholds_df.T
    #ROC曲線の描写
    plt.plot(fpr,tpr,label='roc curve (AUC = %0.3f)' % auc(fpr,tpr))
    plt.plot([0,0,1], [0,1,1], linestyle='--', label='ideal line')
    plt.plot([0, 1], [0, 1], linestyle='--', label='random prediction')
    plt.legend()
    plt.xlabel('false positive rate(FPR)')
    plt.ylabel('true positive rate(TPR)')
    plt.show()
    return acc.item(),test_loss
  

# 訓練のクラスを用意する
def train_net(net,train_loader,
              test_loader,
              only_fc=True,
              optimizer_cls = optim.Adam,
              loss_fn = nn.CrossEntropyLoss(),
              n_iter=10,device="cpu"):
    train_losses =[]
    test_losses=[]
    train_loss=[]
    test_loss=[]
    train_acc =[]
    val_acc = []
    if only_fc:
        #最後の線形層のパラメータのみをoptimizerに渡す
        optimizer = optimizer_cls(net.fc.parameters())
    else:
        optimizer=optimizer_cls(net.parameters())
    for epoch in range(n_iter):
        running_loss=0.0
        #ネットワークを訓練モードにする
        net.train()
        n=0
        n_acc=0
        #tqdmを使用してプログレスバーを出す
        for i,(xx,yy) in tqdm.tqdm(enumerate(train_loader),
                                   total=len(train_loader)):
            xx=xx.to(device)
            yy=yy.to(device)
            h=net(xx)
            loss=loss_fn(h,yy).float()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
            n+=len(xx)
            _,y_pred=h.max(1)
            n_acc+=(yy == y_pred).float().sum().item()
        train_losses.append(running_loss/i)
        print("train_losses=",train_losses)
        #訓練データの予測精度
        train_acc.append(n_acc/n)
        #検証データの予測精度とtestのlossをeval_netクラスから受け取る
        a,test_loss=eval_net(net,test_loader,nn.CrossEntropyLoss(),device)
        val_acc.append(a)
        test_losses.append(test_loss)
        #このepochでの結果を表示
        print("検証回数",epoch,"trainロス",train_losses[-1],"trainの正確さ",train_acc[-1],"testの正確さ",val_acc[-1],flush = True)
    print("最終的な各エポックのtrain_losses=",train_losses,end="\n")
    print("最終的なtest_losses=",test_losses,end="\n")
    # listを転送
    return train_losses,test_losses


#ネットワークの全パラメータをGPUに転送
net.to("cuda:0")
#訓練を実行し、trainとtestのlossをtrain_netから受け取る
train_lossess=[]
test_lossess=[]
train_lossess,test_lossess=train_net(net,train_loader,test_loader,n_iter=20,device = "cuda:0")
#lossグラフの描写
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(train_lossess,label='trainig')
plt.plot(test_lossess,label='test')
plt.title('model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()