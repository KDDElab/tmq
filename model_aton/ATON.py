"""
This script implements an outlier interpretation method of the following paper:
"Beyond Outlier Detection: Outlier Interpretation by Attention-Guided Triplet Deviation Network". in WWW'21.
@ Author: Hongzuo Xu
@ email: hongzuo.xu@gmail.com or leogarcia@126.com or xuhongzuo13@nudt.edu.cn
"""

import threading
import multiprocessing
from loguru import logger
from pyod.models import lscp
import numpy as np
import time, math
import torch.nn as nn
import torch
import torch.utils.data as Data
import torch.optim as optim
import torch.nn.functional as F

from torch.optim import lr_scheduler
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics
from tqdm import tqdm
from model_aton.utils import EarlyStopping, min_max_normalize

from model_aton.datasets import MyHardSingleTripletSelector
from model_aton.datasets import MyHardSingleTripletSelectorOne
from model_aton.datasets import MyHardSingleTripletSelectorTwo

from model_aton.datasets import SingleTripletDataset
from model_aton.datasets import SingleTripletDatasetOne
from model_aton.datasets import SingleTripletDatasetTwo

from model_aton.networks import ATONnet, ATONnet_ablanet, AttentionNet, MyLoss_Q, MyLoss_one, MyLoss_two, MyMultiheadAttention, AttentionNetplus, MyMultiheadAttention_local
from model_aton.networks import MyLoss


# lock = threading.Lock()

class ATON:
    def __init__(self, nbrs_num=10, rand_num=10, nrand_num=1, alpha1=0.8, alpha2=0.2,
                 n_epoch=10, batch_size=64, lr=0.1, n_linear=64, margin=2.,
                 verbose=True, gpu=True):
        self.verbose = verbose

        self.x = None
        self.y = None
        self.ano_idx = None
        self.dim = None

        # a list of normal nbr of each anomaly
        self.normal_nbr_indices = []
        self.anomal_nbr_indices = []

        cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda and gpu else "cpu")
        if cuda:
            torch.cuda.set_device(0)
        print("device:", self.device)

        self.nbrs_num = nbrs_num
        self.rand_num = rand_num
        self.nrand_num = nrand_num
        #
        self.alpha1 = alpha1
        #
        self.alpha2 = alpha2

        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.n_linear = n_linear
        #
        self.margin = margin
        return

    def fit(self, x, y):
        device = self.device
        # x.shape[0]行,x.shape[1]列.
        self.dim = x.shape[1]
        x = min_max_normalize(x)
        #y==1的行索引
        self.ano_idx = np.where(y == 1)[0]

        self.x = torch.tensor(x, dtype=torch.float32).to(device)
        self.y = torch.tensor(y, dtype=torch.int64).to(device)
        self.prepare_nbrs()
        self.prepare_onbrs()

        # train model for each anomaly
        W_lst = []

        if self.verbose:
            iterator = range(len(self.ano_idx))
        else:
            iterator = tqdm(range(len(self.ano_idx)))

   
        for ii in iterator:
            idx = self.ano_idx[ii]

            s_t = time.time()
            W = self.interpret_ano(ii)
            W_lst.append(W)

            if self.verbose:
                print("Ano_id:[{}], ({}/{}) \t time: {:.2f}s\n".format(
                    idx, (ii + 1), len(self.ano_idx),
                    (time.time() - s_t)))

        fea_weight_lst = []
        for ii, idx in enumerate(self.ano_idx):
            w = W_lst[ii]

            fea_weight = np.zeros(self.dim)
            # attention (linear space) + w --> feature weight (original space)
            for j in range(len(w)):
                fea_weight += abs(w[j])
            fea_weight_lst.append(fea_weight)
        return fea_weight_lst
        

    def interpret_ano(self, ii, device=None):
        idx = self.ano_idx[ii]
        #if device is None:
        device = self.device
        dim = self.dim

        nbr_indices = self.normal_nbr_indices[ii]
        onbr_indices = self.anomal_nbr_indices[ii]
        data_loader, test_loader = self.prepare_triplets(idx, nbr_indices, onbr_indices)
        n_linear = self.n_linear
      
        model = ATONnet(n_feature=dim, n_linear=n_linear, margin=self.margin, alpha1=self.alpha1, alpha2=self.alpha2)
        
        model.to(device)
        # print(model)
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-2)
        #criterion = MyLoss(alpha1=self.alpha1, alpha2=self.alpha2, margin=self.margin)

        #每训练5个epoch，更新一次参数
        scheduler = lr_scheduler.StepLR(optimizer, 5, gamma=0.1)
        #早停法，当有连续的patience个轮次数值没有继续下降，反而上升的时候结束训练的条件
        early_stp = EarlyStopping(patience=3, verbose=False)


        # logger.info(self.n_epoch)
        
        for epoch in range(self.n_epoch):
            #作用是 启用 batch normalization 和 dropout 。
            model.train()
            total_loss = 0
            total_dis = 0
            es_time = time.time()

            # logger.info("start")
            batch_cnt = 0
            # logger.info("{} ----- start".format(ii))
            for anchor, pos, neg, nneg in data_loader:
                
                # print(anchor.shape)
                anchor, pos, neg, nneg = anchor.to(device), pos.to(device), neg.to(device), nneg.to(device)
                #embed_anchor, embed_pos, embed_neg, embed_nneg, attn, dis = model(anchor, pos, neg, nneg)
                #loss = criterion(embed_anchor, embed_pos, embed_neg, embed_nneg, dis)
                loss = model(anchor, pos, neg, nneg)

                # loss = loss.sum()

                total_loss += loss

                optimizer.zero_grad()
                # print(loss.shape)
                loss.backward()
                optimizer.step()
                batch_cnt += 1
            # logger.info("{} ----- end".format(ii))
            # logger.info("end")
            train_loss = total_loss / batch_cnt
            est = time.time() - es_time

            if self.verbose and (epoch + 1) % 1 == 0:
                message = 'Epoch: [{:02}/{:02}]  loss: {:.4f} Time: {:.2f}s'.format(epoch + 1, self.n_epoch,
                                                                                    train_loss, est)
                print(message)
            scheduler.step()

            # lock.acquire()
            early_stp(train_loss, model)
            # lock.release()

            if early_stp.early_stop:
                # lock.acquire()
                model.load_state_dict(torch.load(early_stp.path))
                # lock.release()
                if self.verbose:
                    print("early stopping")
                break
        # distill W and attn from network
        for anchor, pos, neg, nneg in test_loader:
            model.eval()
            anchor, pos, neg, nneg = anchor.to(device), pos.to(device), neg.to(device), nneg.to(device)
            _= model(anchor, pos, neg, nneg)
            #_, _, _, _, attn, _ = model(anchor, pos, neg, nneg)
        W = model.linear.weight.data.cpu().numpy()
        return W



    def prepare_triplets(self, idx, nbr_indices, onbr_indices):
        x = self.x
        y = self.y
        selector = MyHardSingleTripletSelector(nbrs_num=self.nbrs_num, rand_num=self.rand_num, nrand_num=self.nrand_num, 
                                               nbr_indices=nbr_indices, onbr_indices = onbr_indices)
        dataset = SingleTripletDataset(idx, x, y, triplets_selector=selector)
        data_loader = Data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = Data.DataLoader(dataset, batch_size=len(dataset))
        return data_loader, test_loader


    def prepare_nbrs(self):
        x = self.x.cpu().data.numpy()
        y = self.y.cpu().data.numpy()

        anom_idx = np.where(y == 1)[0]
        x_anom = x[anom_idx]
        noml_idx = np.where(y == 0)[0]
        x_noml = x[noml_idx]
        n_neighbors = self.nbrs_num

        nbrs_local = NearestNeighbors(n_neighbors=n_neighbors).fit(x_noml)
        tmp_indices = nbrs_local.kneighbors(x_anom)[1]

        for idx in tmp_indices:
            nbr_indices = noml_idx[idx]
            self.normal_nbr_indices.append(nbr_indices)
        return
    
    def prepare_onbrs(self):
        x = self.x.cpu().data.numpy()
        y = self.y.cpu().data.numpy()

        anom_idx = np.where(y == 1)[0]
        
        x_nanom = x[anom_idx]

        on_neighbors = self.nrand_num
        onbrs_local = NearestNeighbors(n_neighbors=on_neighbors).fit(x_nanom)
        ntmp_indices = onbrs_local.kneighbors(x_nanom)[1]

        for idx in ntmp_indices:
            onbr_indices = anom_idx[idx]
            self.anomal_nbr_indices.append(onbr_indices)
        return

