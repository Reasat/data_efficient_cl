import os
import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from tqdm import tqdm
import lightly
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize,StandardScaler
from sklearn import metrics
from PIL import Image
import numpy as np
import time
import sys
import argparse  
from logger import Logger
import openslide
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import torch
import torch.nn as nn
import torchvision
from lightly.models.modules.heads import SimCLRProjectionHead
from lightly.loss import NTXentLoss
import pytorch_lightning as pl
import random
from datetime import datetime
from scipy.stats import entropy
import sklearn
import pickle
from datamodules import KatherDataset



def print_stat(y, msg = 'printing stats'):
    y = np.array(y)
    print('===============================')
    print(msg)
    print('samples', len(y))
    print('label classes', np.unique(y))
    for item in np.unique(y):
        print('label: {}, count: {}'.format(item, len((y[y==item]))))
    print('===============================')


def linear_evaluation(encoder, classifier, dataloader_train, dataloader_test, max_epoch):
    encoder.eval()
    classifier.train()
    class_num = 2 
    print_stat(dataloader_train.dataset.y, 'proxy train statistics')
    for epoch in range(max_epoch):
        running_loss = 0
        for batch_idx, (batch, label, _) in enumerate(dataloader_train):
#             print(label)
            batch = batch.cuda()
            label = label.cuda()
            with torch.no_grad():
                feat = encoder(batch)
#                 print(feat.shape)
            out = classifier(feat.view(feat.shape[0],feat.shape[1]))
            
            label = torch.nn.functional.one_hot(label,class_num)
            loss = classifier.criterion(out,label.float())
            running_loss+=loss.item()
            classifier.optim.zero_grad()
            loss.backward()
            classifier.optim.step()
#             print(loss.item())
        print('lin_eval: epoch: {}, train_loss: {:.4f}'.format(epoch, running_loss/(batch_idx+1)))
    
    encoder.eval()
    classifier.eval()
    out_all = []
    running_loss = 0
    print('lin_eval: encoding test set and computing metrics ...')
    for batch_idx, (batch, label, _) in enumerate(tqdm(dataloader_test)):
        with torch.no_grad():
            batch = batch.cuda()
            label = label.cuda()
            feat = encoder(batch)
    #                 print(feat.shape)
            out = classifier(feat.view(feat.shape[0],feat.shape[1]))
            out_all.append(out.detach().cpu().numpy())
            label = torch.nn.functional.one_hot(label,class_num)
            loss = classifier.criterion(out, label.float())
            running_loss+=loss.item()

    out_all = np.concatenate(out_all)
    test_loss = running_loss/(batch_idx+1)
    y_preds = np.argmax(out_all, axis=1)
      
    print_stat(dataloader_test.dataset.y, 'test set statistics')
    y = dataloader_test.dataset.y
    acc = sklearn.metrics.accuracy_score(y, y_preds)
    precision = sklearn.metrics.precision_score(y, y_preds)
    recall = sklearn.metrics.recall_score(y, y_preds)
    f1 = sklearn.metrics.f1_score(y, y_preds)
    bal_acc = sklearn.metrics.balanced_accuracy_score(y, y_preds)
    print('lin_eval: test_loss: {:.4f},  acc: {:.4f}, bal_acc: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}'.format(
        test_loss,
        acc,
        bal_acc,
        precision,
        recall,
        f1
    ))
    metrics_dict = {
        'acc': acc,
        'bal_acc': bal_acc,
        'recall': recall,
        'precision' : precision,
        'f1' : f1
    }
    return classifier, out_all, metrics_dict

def k_center_greedy_with_given_centers(dmat,k,given_centers):
    '''Greedy K-center
    INPUT:
    dmat: pairwise distance matrix of size nxn
    k: number of centers, integer smaller than n
    given_centers: integer-vector with entries in 0,...,n-1 as initial centers
    RETURNS: approx. optimal centers
    '''


    n=dmat.shape[0]
#     print(dmat)

    if k==0:
        cluster_centers = np.array([],dtype=int)
    else:
        if given_centers.size==0:
            cluster_centers = np.random.choice(n,1,replace=False)
            kk = 1
        else:
            cluster_centers = given_centers
            kk = 0
#         print(np.ix_(cluster_centers,np.arange(n)))
#         print(dmat[np.ix_(cluster_centers,np.arange(n))])
        distance_to_closest = np.amin(dmat[np.ix_(cluster_centers,np.arange(n))],axis=0)
        while kk<k:
#             print('kk',kk)
#             print('distance_to_closest',distance_to_closest)
            ind_sort = np.argsort(distance_to_closest)[::-1]
            ind_sort = [item for item in ind_sort if item not in cluster_centers]
            temp = ind_sort[0]
#             print('temp',temp)
#             print('dmat[temp,:]',dmat[temp,:])
#             print(np.vstack((distance_to_closest,dmat[temp,:])))
            cluster_centers = np.append(cluster_centers,temp)
            distance_to_closest = np.amin(dmat[np.ix_(cluster_centers,np.arange(n))],axis=0)
#             distance_to_closest = np.amin(np.vstack((distance_to_closest,dmat[temp,:])),axis=0)
#             print('distance_to_closest',distance_to_closest)

            kk+=1
#             print(datetime.now(), kk)

        cluster_centers = cluster_centers[given_centers.size:]
#         print('cluster_centers',cluster_centers)


    return cluster_centers


def make_datasets_dataloaders(paths_train, y_train, paths_test, y_test, args, 
        proxy_sample, train_transforms, test_transforms, collate_fn):
    '''
    
    paths_train --> list of all train filepaths
    paths_finetune --> keep some files for SSL features finetune and rest for train
    paths_pretrain --> select initial simclr pretrain paths from paths_train1
    paths_pool --> paths to select new batch of pretrain paths in each iteration
    paths_proxy --> a small labeled dataset to train proxy model
    
    '''
    paths_train1, paths_finetune, y_train1, y_finetune = train_test_split(
        paths_train,
        y_train,
        test_size= args.finetune_size,
        stratify= y_train,
        random_state= 42
    )
    # created paths finetune
    
    paths_pretrain, paths_pool, y_pretrain, y_pool = train_test_split(
        paths_train1,
        y_train1,
        train_size= args.initial_pretrain_size,
        stratify= y_train1,
        random_state= 42
    )
    # created paths pretrain
    print('train_size', len(paths_pretrain))
    
    paths_proxy, paths_pool , y_proxy, y_pool =train_test_split(
            paths_pool,
            y_pool,
            stratify = y_pool,
            train_size = proxy_sample,
            random_state=42
            )
    
    # created paths proxy and paths pool


    dataset_train_simclr = KatherDataset(
        paths = paths_pretrain,
        y = y_pretrain,
        transforms = None
    )
    dataloader_train_simclr = torch.utils.data.DataLoader(
        dataset_train_simclr,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=args.num_workers
    )
    print_stat(dataset_train_simclr.y, 'ssl train statistics')

    dataset_test = KatherDataset(
        paths = paths_test,
        y = y_test,
        transforms=test_transforms
    )
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers
    )
    print_stat(y_test, 'test statistics')

    dataset_proxy_train = KatherDataset(
                paths = paths_proxy,
                y = y_proxy,
                transforms=train_transforms
            )
    dataloader_proxy_train = torch.utils.data.DataLoader(
            dataset_proxy_train,
            batch_size=args.batch_size_ft,
            shuffle=True,
            drop_last=False,
            num_workers=args.num_workers
        )
    print_stat(y_proxy, 'proxy train statistics')

    dataset_finetune_train = KatherDataset(
                paths = paths_finetune,
                y = y_finetune,
                transforms=train_transforms
            )
    dataloader_finetune_train = torch.utils.data.DataLoader(
            dataset_finetune_train,
            batch_size=args.batch_size_ft,
            shuffle=True,
            drop_last=False,
            num_workers=args.num_workers
        )
    print_stat(y_finetune, 'finetune train statistics')

    return (dataset_train_simclr, dataloader_train_simclr,
            dataset_finetune_train, dataloader_finetune_train,
            dataset_proxy_train, dataloader_proxy_train,
            dataset_test, dataloader_test,
            paths_pool, y_pool)


