# @Author   : Chen Mingyang
# @Time     : 2020/9/2
# @FileName : z_dataloader.py

import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict as ddict
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class TrainDataset(Dataset):
    def __init__(self, triples, nentity, negative_sample_size):
        self.len = len(triples)
        self.triples = triples
        self.nentity = nentity
        self.negative_sample_size = negative_sample_size

        self.hr2t = ddict(set)
        for h, r, t in triples:
            self.hr2t[(h, r)].add(t)
        for h, r in self.hr2t:
            self.hr2t[(h, r)] = np.array(list(self.hr2t[(h, r)]))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        positive_sample = self.triples[idx]
        head, relation, tail = positive_sample

        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size * 2)
            mask = np.in1d(
                negative_sample,
                self.hr2t[(head, relation)],
                assume_unique=True,
                invert=True
            )
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        negative_sample = torch.from_numpy(negative_sample)
        positive_sample = torch.LongTensor(positive_sample)

        return positive_sample, negative_sample, idx

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        sample_idx = torch.tensor([_[2] for _ in data])
        return positive_sample, negative_sample, sample_idx


class TestDataset(Dataset):
    def __init__(self, triples, all_true_triples, nentity, ent_mask=None):
        self.len = len(triples)
        self.triple_set = all_true_triples
        self.triples = triples
        self.nentity = nentity

        self.ent_mask = ent_mask

        self.hr2t_all = ddict(set)
        for h, r, t in all_true_triples:
            self.hr2t_all[(h, r)].add(t)

    def __len__(self):
        return self.len

    @staticmethod
    def collate_fn(data):
        triple = torch.stack([_[0] for _ in data], dim=0)
        trp_label = torch.stack([_[1] for _ in data], dim=0)
        #print(trp_label)
        return triple, trp_label

    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]
        label = self.hr2t_all[(head, relation)]
        trp_label = self.get_label(label)
        triple = torch.LongTensor((head, relation, tail))

        return triple, trp_label

    def get_label(self, label):
        #print(label)
        y = np.zeros([self.nentity], dtype=np.float32)

        if type(self.ent_mask) == np.ndarray:
            y[self.ent_mask] = 1.0

        for e2 in label:
            y[e2] = 1.0
        return torch.FloatTensor(y)

class TestRelDataset(Dataset):
    def __init__(self, triples, all_true_triples, nrelation, ent_mask=None):
        self.len = len(triples)
        self.triple_set = all_true_triples
        self.triples = triples
        self.nrelation = nrelation
        # print("self.nrelation")
        # print(self.nrelation)

        #self.ent_mask = ent_mask

        # self.hr2t_all = ddict(set)
        # for h, r, t in all_true_triples:
        #     self.hr2t_all[(h, r)].add(t)

        self.ht2r_all = ddict(set)
        for h, r, t in all_true_triples:
            self.ht2r_all[(h, t)].add(r)

    def __len__(self):
        return self.len

    @staticmethod
    def collate_fn(data):
        triple = torch.stack([_[0] for _ in data], dim=0)
        trp_label = torch.stack([_[1] for _ in data], dim=0)
        #print(trp_label.shape)
        return triple, trp_label

    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]
        label = self.ht2r_all[(head, tail)]
        trp_label = self.get_label(label)
        #print(trp_label)
        triple = torch.LongTensor((head, relation, tail))

        return triple, trp_label

    def get_label(self, label):
        #print(label)
        y = np.ones([self.nrelation], dtype=np.float32)

        # if type(self.ent_mask) == np.ndarray:
        #     y[self.ent_mask] = 1.0

        # for r in label:
        #     y[r] = 1.0
        return torch.FloatTensor(y)


class TestDataset_Entire(Dataset):
    def __init__(self, triples, all_true_triples, nentity, triple_client_idx=None, ent_mask=None):
        self.len = len(triples)
        self.triple_set = all_true_triples
        self.triples = triples
        self.nentity = nentity

        self.triple_client_idx = torch.tensor(triple_client_idx, dtype=torch.int)
        self.ent_mask = ent_mask

        self.hr2t_all = ddict(set)
        for h, r, t in all_true_triples:
            self.hr2t_all[(h, r)].add(t)

    def __len__(self):
        return self.len

    @staticmethod
    def collate_fn(data):
        triple = torch.stack([_[0] for _ in data], dim=0)
        trp_label = torch.stack([_[1] for _ in data], dim=0)
        triple_idx = torch.stack([_[2] for _ in data], dim=0)
        return triple, trp_label, triple_idx

    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]
        triple_idx = self.triple_client_idx[idx]
        label = self.hr2t_all[(head, relation)]
        trp_label = self.get_label(label, triple_idx)
        triple = torch.LongTensor((head, relation, tail))

        return triple, trp_label, triple_idx

    def get_label(self, label, triple_idx=None):
        y = np.zeros([self.nentity], dtype=np.float32)

        if triple_idx is not None and type(self.ent_mask) == list:
            y[self.ent_mask[triple_idx]] = 1.0

        for e2 in label:
            y[e2] = 1.0
        return torch.FloatTensor(y)


def get_task_dataset(data, args):
    nentity = len(np.unique(data['train']['edge_index'].reshape(-1)))
    nrelation = len(np.unique(data['train']['edge_type']))

    train_triples = np.stack((data['train']['edge_index'][0],
                              data['train']['edge_type'],
                              data['train']['edge_index'][1])).T

    valid_triples = np.stack((data['valid']['edge_index'][0],
                              data['valid']['edge_type'],
                              data['valid']['edge_index'][1])).T

    test_triples = np.stack((data['test']['edge_index'][0],
                             data['test']['edge_type'],
                             data['test']['edge_index'][1])).T

    all_triples = np.concatenate([train_triples, valid_triples, test_triples])
    train_dataset = TrainDataset(train_triples, nentity, args.num_neg)
    valid_dataset = TestDataset(valid_triples, all_triples, nentity)
    test_dataset = TestDataset(test_triples, all_triples, nentity)

    return train_dataset, valid_dataset, test_dataset, nrelation, nentity


def get_task_dataset_entire(data, args):
    train_edge_index = np.array([[], []], dtype=np.int)
    train_edge_type = np.array([], dtype=np.int)

    valid_edge_index = np.array([[], []], dtype=np.int)
    valid_edge_type = np.array([], dtype=np.int)

    test_edge_index = np.array([[], []], dtype=np.int)
    test_edge_type = np.array([], dtype=np.int)

    train_client_idx = []
    valid_client_idx = []
    test_client_idx = []
    client_idx = 0
    for d in data:
        train_edge_index = np.concatenate([train_edge_index, d['train']['edge_index_ori']], axis=-1)
        valid_edge_index = np.concatenate([valid_edge_index, d['valid']['edge_index_ori']], axis=-1)
        test_edge_index = np.concatenate([test_edge_index, d['test']['edge_index_ori']], axis=-1)

        train_edge_type = np.concatenate([train_edge_type, d['train']['edge_type_ori']], axis=-1)
        valid_edge_type = np.concatenate([valid_edge_type, d['valid']['edge_type_ori']], axis=-1)
        test_edge_type = np.concatenate([test_edge_type, d['test']['edge_type_ori']], axis=-1)

        train_client_idx.extend([client_idx] * d['train']['edge_type_ori'].shape[0])
        valid_client_idx.extend([client_idx] * d['valid']['edge_type_ori'].shape[0])
        test_client_idx.extend([client_idx] * d['test']['edge_type_ori'].shape[0])
        client_idx += 1

    nrelation = len(np.unique(train_edge_type))
    nentity = len(np.unique(train_edge_index.reshape(-1)))

    ent_mask = []
    for idx, d in enumerate(data):
        client_mask_ent = np.setdiff1d(np.arange(nentity),
                                       np.unique(d['train']['edge_index_ori'].reshape(-1)), assume_unique=True)
        ent_mask.append(client_mask_ent)

    train_triples = np.stack((train_edge_index[0],
                              train_edge_type,
                              train_edge_index[1])).T
    valid_triples = np.stack((valid_edge_index[0],
                              valid_edge_type,
                              valid_edge_index[1])).T
    test_triples = np.stack((test_edge_index[0],
                             test_edge_type,
                             test_edge_index[1])).T
    all_triples = np.concatenate([train_triples, valid_triples, test_triples])

    train_dataset = TrainDataset(train_triples, nentity, args.num_neg)
    valid_dataset = TestDataset_Entire(valid_triples, all_triples, nentity, valid_client_idx, ent_mask)
    test_dataset = TestDataset_Entire(test_triples, all_triples, nentity, test_client_idx, ent_mask)

    return train_dataset, valid_dataset, test_dataset, nrelation, nentity


def get_all_clients(all_data, args):
    all_ent = np.array([], dtype=int)
    all_rel = np.array([], dtype=int)
    for data in all_data:
        all_ent = np.union1d(all_ent, data['train']['edge_index_ori'].reshape(-1))
        all_rel = np.union1d(all_rel, data['train']['edge_type_ori'].reshape(-1))
    all_ent = np.union1d(all_ent, data['valid']['edge_index_ori'].reshape(-1))
    all_rel = np.union1d(all_rel, data['valid']['edge_type_ori'].reshape(-1))
    all_ent = np.union1d(all_ent, data['test']['edge_index_ori'].reshape(-1))
    all_rel = np.union1d(all_rel, data['test']['edge_type_ori'].reshape(-1))

    
    nentity = len(all_ent)
    print(nentity)

    nrelation = len(all_rel)
    print("nrelation")
    print(nrelation)

    train_dataloader_list = []
    test_dataloader_list = []
    testrel_dataloader_list = []
    valid_dataloader_list = []
    rel_embed_list = []

    ent_freq_list = []

    #all_data 包含n个客户端的数据集
    for data in tqdm(all_data):
        #nrelation = len(np.unique(data['train']['edge_type']))
        #print(nrelation)

        #每行为一个三元组
        train_triples = np.stack((data['train']['edge_index_ori'][0],
                                  data['train']['edge_type'],
                                  data['train']['edge_index_ori'][1])).T
        #print(train_triples)
        valid_triples = np.stack((data['valid']['edge_index_ori'][0],
                                  data['valid']['edge_type'],
                                  data['valid']['edge_index_ori'][1])).T

        test_triples = np.stack((data['test']['edge_index_ori'][0],
                                 data['test']['edge_type'],
                                 data['test']['edge_index_ori'][1])).T


        #setdiff1d 1.功能：找到2个数组中集合元素的差异。

        #2.返回值：在ar1中但不在ar2中的已排序的唯一值。

        client_mask_ent = np.setdiff1d(np.arange(nentity),
                                       np.unique(data['train']['edge_index_ori'].reshape(-1)), assume_unique=True)

        #print(client_mask_ent)

        all_triples = np.concatenate([train_triples, valid_triples, test_triples])
        train_dataset = TrainDataset(train_triples, nentity, args.num_neg)
        valid_dataset = TestDataset(valid_triples, all_triples, nentity, client_mask_ent)
        test_dataset = TestDataset(test_triples, all_triples, nentity, client_mask_ent)
        testrel_dataset = TestRelDataset(test_triples, all_triples, nrelation, client_mask_ent)

        # dataloader
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=TrainDataset.collate_fn
        )
        train_dataloader_list.append(train_dataloader)

        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=args.test_batch_size,
            collate_fn=TestDataset.collate_fn
        )
        valid_dataloader_list.append(valid_dataloader)

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.test_batch_size,
            collate_fn=TestDataset.collate_fn
        )
        test_dataloader_list.append(test_dataloader)


        testrel_dataloader = DataLoader(
            testrel_dataset,
            batch_size=args.test_batch_size,
            collate_fn=TestRelDataset.collate_fn
        )
        testrel_dataloader_list.append(testrel_dataloader)

        embedding_range = torch.Tensor([(args.gamma + args.epsilon) / args.hidden_dim])
        if args.model in ['ComplEx']:
            rel_embed = torch.zeros(nrelation, args.hidden_dim * 2).to(args.gpu).requires_grad_()
        else:
            rel_embed = torch.zeros(nrelation, args.hidden_dim).to(args.gpu).requires_grad_()

        nn.init.uniform_(
            tensor=rel_embed,
            a=-embedding_range.item(),
            b=embedding_range.item()
        )
        rel_embed_list.append(rel_embed)

        # ent_freq  就是 记录节点嵌入是否存在的v
        ent_freq = torch.zeros(nentity)
        # data['train']['edge_index_ori'].reshape(-1) 将[[h][t]]形状的集合变成[]
        for e in data['train']['edge_index_ori'].reshape(-1):
            ent_freq[e] += 1
        ent_freq_list.append(ent_freq)

    ent_freq_mat = torch.stack(ent_freq_list).to(args.gpu)

    return train_dataloader_list, valid_dataloader_list, test_dataloader_list,testrel_dataloader_list, \
           ent_freq_mat, rel_embed_list, nentity
