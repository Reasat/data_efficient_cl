# %%
# Imports
# -------
#
# Import the Python frameworks we need for this tutorial.
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
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import torch
import torch.nn as nn
import torchvision
from lightly.models.modules.heads import SimCLRProjectionHead
from lightly.loss import NTXentLoss
import pytorch_lightning as pl
import random
from datetime import datetime
#import umap.umap_ as umap
from scipy.stats import entropy
import sklearn
import pickle

os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8' #or CUBLAS_WORKSPACE_CONFIG=:16:8
# Let's set the seed for our experiments
def seed_everything(seed = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(seed)

# In[86]:


# %%
# Configuration
# -------------
# 
# We set some configuration parameters for our experiment.
# Feel free to change them and analyze the effect.
#
# The default configuration with a batch size of 256 and input resolution of 128
# requires 6GB of GPU memory.
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type = int, default = 100)
parser.add_argument('--epoch_projection', type = int, default = 0)
parser.add_argument('--iter', type = int, default = 20)
parser.add_argument('--freeze_epochs', type = int)
parser.add_argument('--finetune_epochs', type = int, default = 40)
parser.add_argument('--gpus', default = '1')
parser.add_argument('--batch_size', type = int, default =128)
parser.add_argument('--batch_size_ft', type = int, default = 128)
parser.add_argument('--num_workers', type = int, default =0)
parser.add_argument('--lr_simclr', type = float, default = 0.0001)
parser.add_argument('--lr_projection', type = float, default = 0.0001)
parser.add_argument('--lr_finetune', type = float, default = 0.001)
parser.add_argument('--resize', type = int, default = 224)
parser.add_argument('--finetune_size', type = int, default = 1000)
parser.add_argument('--pretrain_size', type = int, default = 1000)
parser.add_argument('--subset_pool_size', type = int, default = 10000)
parser.add_argument('--new_samples_size', type = int, default = 1000)
parser.add_argument('--save_top_k', type = int, default = -1)
parser.add_argument('--every_n_epochs', type = int, default = 10)
parser.add_argument('--resume_from')
parser.add_argument('--sampling_strategy', default = 'uncertainty')

parser.add_argument('--seed', type = int, default = 0)


# '/home/reasatt/Projects/napari_annotation/model_data/2023-01-13-21-55-21/epoch=499-step=390500-train_loss_ssl=3.19099450.ckpt'
args = parser.parse_args()
seed_everything(args.seed)

class Classifier(torch.nn.Module):
    def __init__(self,lr_finetune):
        super().__init__()
        self.classifier = torch.nn.Linear(512,9)
        self.optim = torch.optim.Adam(self.classifier.parameters(), lr = lr_finetune)
        self.criterion = torch.nn.CrossEntropyLoss()
    def forward(self,x):
        x = self.classifier(x)
        return x
    
class SimCLRModel(pl.LightningModule):
    def __init__(self, max_epochs, lr = 6e-2,  pretrained = False):
        super().__init__()

        # create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18(pretrained = pretrained)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        hidden_dim = resnet.fc.in_features
        self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, 128)

        self.criterion = NTXentLoss(temperature = 0.1)
        self.lr = lr
        self.max_epochs = max_epochs

    def forward(self, x):
        h = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(h)
        return z

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("train_loss_ssl", loss)
        return loss

    def training_step_projection_head(self, batch):
        (x0, x1) = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
#         self.log("train_loss_ssl", loss)
        return loss
    
    def configure_optimizers(self):
        optim = torch.optim.Adam(
            self.parameters(), lr=self.lr, 
#             momentum=0.9, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, self.max_epochs, eta_min=self.lr/10
        )
        return [optim], [scheduler]

class KatherDataset(torch.utils.data.Dataset):
    def __init__(self, paths, y, transforms):
        super().__init__()
        self.paths = paths
        self.y = y
        self.transforms = transforms
    def __len__(self):
        return(len(self.paths))
    def __getitem__(self,idx):
        img = Image.open(self.paths[idx])
        if self.transforms is not None:
            img = self.transforms(img)
        flname = os.path.basename(self.paths[idx])
        return(img, self.y[idx], flname)

def linear_evaluation(encoder, classifier, dataloader_train, dataloader_test, max_epoch):
    encoder.eval()
    classifier.train()
    
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
            
            loss = classifier.criterion(out,label)
            running_loss+=loss.item()
            classifier.optim.zero_grad()
            loss.backward()
            classifier.optim.step()
#             print(loss.item())
    print('epoch: {}, train_loss: {:.4f}'.format(epoch, running_loss/(batch_idx+1)))
    
    encoder.eval()
    classifier.eval()
    out_all = []
    running_loss = 0
    for batch_idx, (batch, label, _) in enumerate(tqdm(dataloader_test)):
        with torch.no_grad():
            batch = batch.cuda()
            label = label.cuda()
            feat = encoder(batch)
    #                 print(feat.shape)
            out = classifier(feat.view(feat.shape[0],feat.shape[1]))
            out_all.append(out.detach().cpu().numpy())
            loss = classifier.criterion(out, label)
            running_loss+=loss.item()

    out_all = np.concatenate(out_all)
    test_loss = running_loss/(batch_idx+1)
    y_preds = np.argmax(out_all, axis=1)
    acc = sklearn.metrics.accuracy_score(dataset_test.y, y_preds)
    precision = sklearn.metrics.precision_score(dataset_test.y, y_preds, average = 'macro')
    recall = sklearn.metrics.recall_score(dataset_test.y, y_preds, average = 'macro')
    f1 = sklearn.metrics.f1_score(dataset_test.y, y_preds, average = 'macro')
    bal_acc = sklearn.metrics.balanced_accuracy_score(dataset_test.y, y_preds)
    print('test_loss: {:.4f},  acc: {:.4f}, bal_acc: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}'.format(
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

#=============================================================================================================================
    
TIME_STAMP=time.strftime('%Y-%m-%d-%H-%M-%S')
dir_output = os.path.join('model_data', TIME_STAMP)
os.makedirs(dir_output, exist_ok = True)

sys.stdout = Logger(os.path.join(dir_output,TIME_STAMP+'.log'))

print(TIME_STAMP)

for arg in vars(args):
    print('{}: {}'.format(arg, getattr(args, arg)))
    
    filepath_cfg = os.path.join(dir_output, TIME_STAMP+'.cfg')
    with open(filepath_cfg,'w') as f:
        for arg in vars(args):
            f.write('{}: {}\n'.format(arg, getattr(args, arg)))
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus 
gpus=len(args.gpus.split(','))

num_workers = args.num_workers
batch_size = args.batch_size
max_epochs = args.epoch
input_size = args.resize
num_ftrs = 32

# %%

# %%
# Make sure `path_to_data` points to the downloaded clothing dataset.
# You can download it using 
# `git clone https://github.com/alexeygrigorev/clothing-dataset.git`
# path_to_data = '/tank/data/NCRF/PATCHES_TRAIN_25000x2'


# In[89]:


from glob import glob


# In[90]:


paths = glob('/tank/mirror/kather-19/NCT-CRC-HE-100K/*/*')


# In[91]:


labels = [p.split(os.sep)[-2] for p in paths]


# In[92]:


CLASSES = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']

# In[93]:


class2category = {c: 1 if c=='TUM' else 0 for i,c in enumerate(CLASSES)}
# In[94]:


y_true = [class2category[l] for l in labels]


# In[95]:


from sklearn.model_selection import train_test_split


# In[96]:


paths_train1, paths_finetune, y_train1, y_finetune = train_test_split(
    paths,
    y_true,
    test_size= args.finetune_size,
    stratify= y_true,
    random_state= 42
)
# resampling train
paths_pretrain, paths_pool, y_pretrain, y_pool = train_test_split(
    paths_train1,
    y_train1,
    train_size= args.pretrain_size,
    stratify= y_train1,
    random_state= 42
)
print('train_size', len(paths_pretrain))
# In[97]:


paths_test = glob('/tank/mirror/kather-19/CRC-VAL-HE-7K/*/*')
labels_test = [p.split(os.sep)[-2] for p in paths_test]
y_test = [class2category[l] for l in labels_test]


# In[98]:


len(paths_test)

# In[100]:


collate_fn = lightly.data.SimCLRCollateFunction(
    input_size=input_size,
    vf_prob=0.5,
    rr_prob=0.5
)

# We create a torchvision transformation for embedding the dataset after 
# training
test_transforms = torchvision.transforms.Compose([
#     torchvision.transforms.Resize((input_size, input_size)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=lightly.data.collate.imagenet_normalize['mean'],
        std=lightly.data.collate.imagenet_normalize['std'],
    )
])

dataset_train_simclr = KatherDataset(
    paths = paths_pretrain,
    y = y_pretrain,
    transforms = None
)
dataloader_train_simclr = torch.utils.data.DataLoader(
    dataset_train_simclr,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    drop_last=True,
    num_workers=num_workers
)
print('simclr train samples',len(dataset_train_simclr))

dataset_test = KatherDataset(
    paths = paths_test,
    y = y_test,
    transforms=test_transforms
)
dataloader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

# finetune
train_transforms = torchvision.transforms.Compose([
#     torchvision.transforms.Resize((input_size, input_size)),
    torchvision.transforms.RandomVerticalFlip(),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=lightly.data.collate.imagenet_normalize['mean'],
        std=lightly.data.collate.imagenet_normalize['std'],
    )
])
dataset_finetune = KatherDataset(
        paths = paths_finetune,
        y = y_finetune,
        transforms=train_transforms
    )

dataloader_finetune = torch.utils.data.DataLoader(
        dataset_finetune,
        batch_size=args.batch_size_ft,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

# Load simclr model

if args.resume_from is not None:
    if args.resume_from == 'imagenet':
        model = SimCLRModel(
        lr= args.lr_simclr,
        max_epochs = args.epoch,
        pretrained=True
    )
    else:
        model = SimCLRModel(
        lr= args.lr_simclr,
        max_epochs = args.epoch,
        pretrained=False
        )
        
        model.load_state_dict(torch.load(args.resume_from)['state_dict'])
    print('loading weights from', args.resume_from)
else:
    model = SimCLRModel(
        lr= args.lr_simclr,
        max_epochs = args.epoch,
        pretrained=False
        )
model = model.cuda()

metrics_dict={}
for i_iter in range(args.iter):
    print('========================================================================')
    print('iteration', i_iter)
    seed_everything(args.seed)
    
    if args.epoch > 0:        
        if args.epoch_projection>0:
            print('training projection head ...')
            model.train()
            optim = torch.optim.Adam(model.projection_head.parameters(),args.lr_finetune)
            # freeze encoder layers
            for param in model.backbone.parameters():
                param.requires_grad=False
            for epoch in range(args.epoch_projection):
                running_loss=0
                for batch_idx, batch in enumerate(dataloader_train_simclr):
#                     print(batch[1])
                    img_batch_1 = batch[0][0].cuda()
                    img_batch_2 = batch[0][1].cuda()
                    optim.zero_grad()
                    loss = model.training_step_projection_head(
                        (img_batch_1,img_batch_2),
                    )
                    running_loss+=loss.item()
                    loss.backward()
                    optim.step()
                print('epoch_projection: {}, head_loss: {:.4f}'.format(epoch,running_loss/(batch_idx+1)))
            
            # thaw encoder layers
            for param in model.enoder.parameters():
                param.requires_grad=True
            
                
                    
        
        print('train simclr...')
        
        dir_output_iter = os.path.join(dir_output,'iter_{}'.format(i_iter))
        os.makedirs(dir_output_iter)
        lr_monitor = LearningRateMonitor(logging_interval="step")
        mc = ModelCheckpoint(
                dirpath=dir_output_iter,
                filename='{epoch}-{step}-{train_loss_ssl:.8f}',
                every_n_epochs = args.every_n_epochs,
                verbose=True,
                monitor="train_loss_ssl",
#                 log_every_n_steps=1,
                save_last=True,
                save_top_k = args.save_top_k,
                save_weights_only = True,
            )
        callbacks = [mc,lr_monitor]

        trainer = pl.Trainer(
        max_epochs=args.epoch, gpus=gpus,
        default_root_dir = dir_output_iter,
        callbacks = callbacks
        )
        model.train()
        trainer.fit(model, dataloader_train_simclr)
    print('linear evaluation...')   
    seed_everything(args.seed)
    classifier = Classifier(args.lr_finetune)
    encoder = model.backbone
    encoder.cuda()
    classifier.cuda()
    metrics_list = []
    for i in tqdm(range(args.finetune_epochs)):
        classifier,out_all, metrics = linear_evaluation(encoder, classifier, dataloader_finetune, dataloader_test, 1)
        y_pred = np.argmax(out_all, axis = 1)
        print('balanced_accuracy_score', sklearn.metrics.balanced_accuracy_score(dataset_test.y, y_pred))
        metrics_list.append(metrics)
    metrics_dict['iter_{}'.format(i_iter)] = metrics_list

    ## resampling procedure
    print('selecting samples from pool...')
    print('pool size before sampling', len(paths_pool))
    
    if args.sampling_strategy == 'random':
        print('using random sampling')
        sampled_ind = np.random.choice(
            range(len(paths_pool)),
            args.new_samples_size,
            replace = False)
        
    ## shrinking pool
    ind_subset = np.random.choice(
            range(len(paths_pool)),
            args.subset_pool_size,
            replace = False
        )
    dataset_pool = KatherDataset(
            paths = np.array(paths_pool)[ind_subset],
            y = np.array(y_pool)[ind_subset],
            transforms=test_transforms
        )
    dataloader_pool = torch.utils.data.DataLoader(
        dataset_pool,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )
    print('size of pool subset', len(dataset_pool))
    
    if args.sampling_strategy == 'k-center':
        print('using k-center strategy')
        encoder.eval()
        classifier.eval()
        out_all = []
        features = []
        for batch_idx, (batch, label, _) in enumerate(tqdm(dataloader_pool)):
    #         print(label)
            with torch.no_grad():
                batch = batch.cuda()
                label = label.cuda()
                feat = encoder(batch)
                feat = feat.view(feat.shape[0],feat.shape[1])
                features.append(feat.detach().cpu().numpy())
        features = np.concatenate(features)
        
#         print('reducing dimensionality')
#         reducer = umap.UMAP(n_neighbors=50,
#                 min_dist=0.5,
#                 n_components=32
#                )
#         features = StandardScaler().fit_transform(features)
#         features = reducer.fit_transform(features)
        
        print(datetime.now(), 'computing distance ...')
        similarity = sklearn.metrics.pairwise_distances(
                    features
                )   
        print(datetime.now(), 'computing k centers ...')
        sampled_ind = k_center_greedy_with_given_centers(
            similarity,
            k=args.new_samples_size,
            given_centers=np.array([])
        )
        sampled_ind = ind_subset[sampled_ind]
        
    
       
    if args.sampling_strategy == 'uncertainty':
        print('using uncertainty based sampling')
        
        encoder.eval()
        classifier.eval()
        out_all = []
        for batch_idx, (batch, label, _) in enumerate(tqdm(dataloader_pool)):
    #         print(label)
            with torch.no_grad():
                batch = batch.cuda()
                label = label.cuda()
                feat = encoder(batch)
                out = classifier(feat.view(feat.shape[0],feat.shape[1]))
                out_all.append(out.detach().cpu().numpy())
        out_all = np.concatenate(out_all)
        # print(out_all)
        out_all = torch.softmax(torch.tensor(out_all), axis = 1)
        out_all = np.array(out_all)
        selection_criteria = entropy(out_all, axis = 1)
        ind_sorted = np.argsort(selection_criteria)[::-1]
        sampled_ind = ind_sorted[:args.new_samples_size]
        sampled_ind = ind_subset[sampled_ind]
        uncertainty_sel = selection_criteria[ind_sorted[:args.new_samples_size]]
        print('mean uncertainty of selected samples', np.mean(uncertainty_sel))
    
    if args.sampling_strategy == 'reverse_uncertainty':
        print('using uncertainty based sampling')
        
        encoder.eval()
        classifier.eval()
        out_all = []
        for batch_idx, (batch, label, _) in enumerate(tqdm(dataloader_pool)):
    #         print(label)
            with torch.no_grad():
                batch = batch.cuda()
                label = label.cuda()
                feat = encoder(batch)
                out = classifier(feat.view(feat.shape[0],feat.shape[1]))
                out_all.append(out.detach().cpu().numpy())
        out_all = np.concatenate(out_all)
        # print(out_all)
        out_all = torch.softmax(torch.tensor(out_all), axis = 1)
        out_all = np.array(out_all)
        selection_criteria = entropy(out_all, axis = 1)
        ind_sorted = np.argsort(selection_criteria)
        sampled_ind = ind_sorted[:args.new_samples_size]
        sampled_ind = ind_subset[sampled_ind]
        uncertainty_sel = selection_criteria[ind_sorted[:args.new_samples_size]]
        print('mean uncertainty of selected samples', np.mean(uncertainty_sel))

    

    paths_new  = np.array(paths_pool)[sampled_ind]
#     with open(dir_output_iter+'/paths_new.txt', 'w') as f:
#         for p,u in zip(paths_new,uncertainty_sel):
#             f.write('{}, {}\n'.format(p,u))
            
    y_new = np.array(y_pool)[sampled_ind]

    paths_pretrain = list(paths_pretrain)+list(paths_new)
    y_pretrain = list(y_pretrain)+list(y_new)

    ind_pool = [ind not in sampled_ind for ind in range(len(paths_pool))]
    paths_pool = np.array(paths_pool)[ind_pool]
    y_pool = np.array(y_pool)[ind_pool]


    dataset_train_simclr = KatherDataset(
        paths = paths_pretrain,
        y = y_pretrain,
        transforms = None
    )
    print('simclr renewed train samples',len(dataset_train_simclr))

    dataloader_train_simclr = torch.utils.data.DataLoader(
        dataset_train_simclr,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=num_workers
    )
    print('remaining pool samples', len(paths_pool))
with open('model_data/{}/{}_metrics.bin'.format(TIME_STAMP,TIME_STAMP), 'wb') as pfile:
    pickle.dump(metrics_dict,pfile)
