
import os
import random
import pickle
import math

import torch
from sklearn.decomposition import PCA

emb_fold = "visual_constrained_test_embedding"
emb_files = os.listdir(emb_fold)
emb_files = sorted(emb_files)

test_embs = []
for emb_file in emb_files:
    with open(os.path.join(emb_fold, emb_file), "rb") as f:
        test_embs.append(pickle.load(f))
'''
random.seed(0)
random.shuffle(all_emb)
train_len = int(len(all_emb)*0.8)
train_emb = all_emb[:train_len]
test_emb = all_emb[int(len(all_emb)*0.8):]

train_emb = torch.stack(train_emb)
margin_ = train_emb[:, 2, ...].contiguous()
train_emb = train_emb[:, 0, ...] - train_emb[:, 2, ...]
mag = torch.norm(train_emb, dim=-1)
margin_ = torch.einsum('bpd,bpd->bp', margin_, train_emb) / mag
margin_ = margin_.mean()
train_emb = train_emb.view(-1, train_emb.shape[-1])
'''
train_direction_file = 'directions/cocoval2017_minigpt4_direction.pkl'
direction = pickle.load(open(train_direction_file,'rb'))
mag = torch.norm(direction)

projections = []
total, acc1, acc2, ab_acc = 0, 0, 0, 0
ab1, metric1 = 0, 0
metric2 = 0
for t_emb in test_embs:
    projection = torch.einsum('bpd,d->bp', t_emb, direction) / mag
    projection = projection.mean(dim=-1)
    if projection[0] > 0: # projection[2]:
        acc1 += 1
    if 0  > projection[2]:
        acc2 += 1
    if 0 > projection[1]:
        ab_acc += 1
    print(projection)
    total += 1
print(f"test size: {len(test_embs)}, margin: 0, att_acc: {acc1/total * 100:.2f}%, raw_acc:{acc2/total * 100:.2f}%, noise_acc:{ab_acc/total * 100:.2f}%")
'''
test_emb = pickle.load(open('adversarial_images/image_embeddings.pkl', 'rb')).squeeze(1)
projection = torch.einsum('bpd,d->bp', test_emb, direction) / mag
projection = projection.mean(dim=-1)
print(projection)
'''
