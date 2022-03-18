import torch
import numpy as np
from collections import defaultdict
from model import PathCon
from utils import sparse_to_tuple
from kgemodel.kgemodel import KGEModel,KGERelModel
from kgemodel.dataloader import TrainDataset,TestRelDataset,TestDataset
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict as ddict
from torch import optim
import torch.nn.functional as F
import torch.nn as nn
import scipy.sparse as sp
import numpy as np

args = None


def train(model_args, data):
    global args, model, sess
    args = model_args

    # extract data
    triplets, paths, n_relations, neighbor_params, path_params = data

    train_triplets, valid_triplets, test_triplets = triplets
    train_edges = torch.LongTensor(np.array(range(len(train_triplets)), np.int32))
    train_entity_pairs = torch.LongTensor(np.array([[triplet[0], triplet[1]] for triplet in train_triplets], np.int32))
    valid_entity_pairs = torch.LongTensor(np.array([[triplet[0], triplet[1]] for triplet in valid_triplets], np.int32))
    test_entity_pairs = torch.LongTensor(np.array([[triplet[0], triplet[1]] for triplet in test_triplets], np.int32))

    train_paths, valid_paths, test_paths = paths

    #标签为r
    train_labels = torch.LongTensor(np.array([triplet[2] for triplet in train_triplets], np.int32))
    valid_labels = torch.LongTensor(np.array([triplet[2] for triplet in valid_triplets], np.int32))
    test_labels = torch.LongTensor(np.array([triplet[2] for triplet in test_triplets], np.int32))

    # define the model
    
    model = PathCon(args, n_relations, neighbor_params, path_params)
    #print(model.state_dict())

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.lr,
        # weight_decay=args.l2,
    )

    if args.cuda:
        model = model.cuda()
        train_labels = train_labels.cuda()
        valid_labels = valid_labels.cuda()
        test_labels = test_labels.cuda()
        if args.use_context:
            train_edges = train_edges.cuda()
            train_entity_pairs = train_entity_pairs.cuda()
            valid_entity_pairs = valid_entity_pairs.cuda()
            test_entity_pairs = test_entity_pairs.cuda()

    # prepare for top-k evaluation
    true_relations = defaultdict(set)
    for head, tail, relation in train_triplets + valid_triplets + test_triplets:
        true_relations[(head, tail)].add(relation)
    best_valid_acc = 0.0
    final_res = None  # acc, mrr, mr, hit1, hit3, hit5

    print('start training ...')

    for step in range(args.epoch):

        # shuffle training data
        index = np.arange(len(train_labels))
        np.random.shuffle(index)
        if args.use_context:
            train_entity_pairs = train_entity_pairs[index]
            train_edges = train_edges[index]
        if args.use_path:
            train_paths = train_paths[index]
        train_labels = train_labels[index]

        # training
        s = 0
        while s + args.batch_size <= len(train_labels):
            loss = model.train_step(model, optimizer, get_feed_dict(
                train_entity_pairs, train_edges, train_paths, train_labels, s, s + args.batch_size))
            s += args.batch_size

        # evaluation
        print('epoch %2d   ' % step, end='')
        train_acc, _ = evaluate(train_entity_pairs, train_paths, train_labels)
        valid_acc, _ = evaluate(valid_entity_pairs, valid_paths, valid_labels)
        test_acc, test_scores = evaluate(test_entity_pairs, test_paths, test_labels)

        # show evaluation result for current epoch
        current_res = 'acc: %.4f' % test_acc
        print('train acc: %.4f   valid acc: %.4f   test acc: %.4f' % (train_acc, valid_acc, test_acc))
        mrr, mr, hit1, hit3, hit5 = calculate_ranking_metrics(test_triplets, test_scores, true_relations)
        current_res += '   mrr: %.4f   mr: %.4f   h1: %.4f   h3: %.4f   h5: %.4f' % (mrr, mr, hit1, hit3, hit5)
        print('           mrr: %.4f   mr: %.4f   h1: %.4f   h3: %.4f   h5: %.4f' % (mrr, mr, hit1, hit3, hit5))
        print()

        # update final results according to validation accuracy
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            final_res = current_res

    # show final evaluation result
    print('final results\n%s' % final_res)


def train_fed(model_args, data,round,client_id,state_dict=None):
    global args, model, sess
    args = model_args
    client_id = client_id

    # extract data
    triplets, paths, n_relations, path_params,fed_data,valid_test_data = data

    _, valid_triplets, test_triplets = triplets

    train_triplets = fed_data[0][client_id]

    local_length = len(train_triplets)

    #本地训练集 全局测试
    # train_triplets = fed_data[0][client_id]
    # neighbor_params = fed_data[2][client_id]
    # train_paths = fed_data[3][client_id]
    # #print(type(train_paths))
    # train_edges = fed_data[4][client_id]
    # train_entity_pairs = fed_data[5][client_id]


    #本地训练集 本地测试
    train_triplets = fed_data[0][client_id][:int(local_length*0.8)]
    neighbor_params = fed_data[2][client_id]
    train_paths = fed_data[3][client_id][:int(local_length*0.8)]
    #print(type(train_paths))
    train_edges = fed_data[4][client_id][:int(local_length*0.8)]
    train_entity_pairs = fed_data[5][client_id][:int(local_length*0.8)]


    valid_entity_pairs = valid_test_data[0]
    # 全局测试集
    test_entity_pairs = valid_test_data[2]

    _, valid_paths, test_paths = paths

    #标签为r
    train_labels = fed_data[6][client_id][:int(local_length*0.8)]
    valid_labels = valid_test_data[1]
    test_labels = valid_test_data[3]

    #修改测试集 这个修改无效
    # test_triplets = test_triplets+fed_data[0][client_id][int(local_length*0.8):]
    # test_entity_pairs =torch.cat((test_entity_pairs,fed_data[5][client_id][int(local_length*0.8):]),0)
    # test_labels = torch.cat((test_labels,fed_data[6][client_id][int(local_length*0.8):]),0)
    # #print(test_paths.toarray().shape)
    # test_paths = sp.lil_matrix(np.vstack((test_paths.toarray(),fed_data[3][client_id][int(local_length*0.8):].toarray())))

    # 有效的本地测试集设置
    test_triplets = fed_data[0][client_id][int(local_length*0.8):]
    test_entity_pairs =fed_data[5][client_id][int(local_length*0.8):]
    test_labels = fed_data[6][client_id][int(local_length*0.8):]
    #print(test_paths.toarray().shape)
    test_paths = fed_data[3][client_id][int(local_length*0.8):]

    # define the model
    
    model = PathCon(args, n_relations, neighbor_params, path_params,client_id+1)
    if state_dict != None:
        model.load_state_dict(state_dict)
    #print(model.state_dict())
    

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.lr,
        # weight_decay=args.l2,
    )

    if args.cuda:
        model = model.cuda()
        train_labels = train_labels.cuda()
        valid_labels = valid_labels.cuda()
        test_labels = test_labels.cuda()
        if args.use_context:
            train_edges = train_edges.cuda()
            train_entity_pairs = train_entity_pairs.cuda()
            valid_entity_pairs = valid_entity_pairs.cuda()
            test_entity_pairs = test_entity_pairs.cuda()

    # prepare for top-k evaluation
    #集合为值的字典
    #true_relations (head,tail)可能存在的relation
    true_relations = defaultdict(set)
    for head, tail, relation in train_triplets + valid_triplets + test_triplets:
        true_relations[(head, tail)].add(relation)
    best_valid_acc = 0.0
    final_res = None  # acc, mrr, mr, hit1, hit3, hit5

    print('round '+str(round)+': client '+str(client_id)+' start training ...'+'dataset: '+args.dataset)

    for step in range(args.fed_epoch):
        #print(model.relation_features)

        # shuffle training data
        index = np.arange(len(train_labels))
        np.random.shuffle(index)
        if args.use_context:
            train_entity_pairs = train_entity_pairs[index]
            train_edges = train_edges[index]
        if args.use_path:
            train_paths = train_paths[index]
        train_labels = train_labels[index]

        # training
        s = 0
        while s + args.batch_size <= len(train_labels):
            loss = model.train_step(model, optimizer, get_feed_dict(
                train_entity_pairs, train_edges, train_paths, train_labels, s, s + args.batch_size))
            s += args.batch_size

        print(loss)
            #临时中断 记得删除
            #break

        # evaluation
        print('epoch %2d   ' % step, end='')
        train_acc, _ = evaluate(train_entity_pairs, train_paths, train_labels)
        valid_acc, _ = evaluate(valid_entity_pairs, valid_paths, valid_labels)
        test_acc, test_scores = evaluate(test_entity_pairs, test_paths, test_labels)

        # show evaluation result for current epoch
        current_res = 'acc: %.4f' % test_acc
        #print('train acc: %.4f   valid acc: %.4f   test acc: %.4f' % (train_acc, valid_acc, test_acc))
        mrr, mr, hit1, hit3, hit5 = calculate_ranking_metrics(test_triplets, test_scores, true_relations)
        current_res += '   mrr: %.4f   mr: %.4f   h1: %.4f   h3: %.4f   h5: %.4f' % (mrr, mr, hit1, hit3, hit5)
        #print('           mrr: %.4f   mr: %.4f   h1: %.4f   h3: %.4f   h5: %.4f' % (mrr, mr, hit1, hit3, hit5))
        print()

        # update final results according to validation accuracy
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            final_res = current_res

    # show final evaluation result
    print('round '+str(round)+': client '+str(client_id)+' final results\n%s' % final_res)
    return model.state_dict()

def get_feed_dict(entity_pairs, train_edges, paths, labels, start, end):
    feed_dict = {}

    if args.use_context:
        feed_dict["entity_pairs"] = entity_pairs[start:end]
        if train_edges is not None:
            feed_dict["train_edges"] = train_edges[start:end]
        else:
            # for evaluation no edges should be masked out
            feed_dict["train_edges"] = torch.LongTensor(np.array([-1] * (end - start), np.int32)).cuda() if args.cuda \
                        else torch.LongTensor(np.array([-1] * (end - start), np.int32))

    if args.use_path:
        if args.path_type == 'embedding':
            indices, values, shape = sparse_to_tuple(paths[start:end])
            #print(shape)
            indices = torch.LongTensor(indices).cuda() if args.cuda else torch.LongTensor(indices)
            values = torch.Tensor(values).cuda() if args.cuda else torch.Tensor(values)
            feed_dict["path_features"] = torch.sparse.FloatTensor(indices.t(), values, torch.Size(shape)).to_dense()
        elif args.path_type == 'rnn':
            feed_dict["path_ids"] = torch.LongTensor(paths[start:end]).cuda() if args.cuda \
                    else torch.LongTensor(paths[start:end])

    feed_dict["labels"] = labels[start:end]

    return feed_dict


def evaluate(entity_pairs, paths, labels):
    acc_list = []
    scores_list = []

    s = 0
    while s + args.batch_size <= len(labels):
        acc, scores = model.test_step(model, get_feed_dict(
            entity_pairs, None, paths, labels, s, s + args.batch_size))
        #print(len(scores)) 237
        #break
        acc_list.extend(acc)
        scores_list.extend(scores)
        s += args.batch_size


    return float(np.mean(acc_list)), np.array(scores_list)


def calculate_ranking_metrics(triplets, scores, true_relations):
    #print(scores.shape) (20352,237)
    for i in range(scores.shape[0]):
        head, tail, relation = triplets[i]
        for j in true_relations[head, tail] - {relation}:
            #print(scores[i,j])
            scores[i, j] -= 1.0

    sorted_indices = np.argsort(-scores, axis=1)
    #argsort函数返回的是数组值从小到大的索引值
    #print(sorted_indices)
    relations = np.array(triplets)[0:scores.shape[0], 2]
    #print(relations.shape)
    sorted_indices -= np.expand_dims(relations, 1)
    #print(sorted_indices)
    #np.argwhere返回非0的数组元组的索引，其中a是要索引数组的条件。
    #返回正确标签在预测序列中的排序
    zero_coordinates = np.argwhere(sorted_indices == 0)
    #print(zero_coordinates)
    # [[    0     0]
    # [    1     0]
    # [    2     0]
    # ...
    # [20349     0]
    # [20350     0]
    # [20351     0]]
    rankings = zero_coordinates[:, 1] + 1

    mrr = float(np.mean(1 / rankings))
    mr = float(np.mean(rankings))
    hit1 = float(np.mean(rankings <= 1))
    hit3 = float(np.mean(rankings <= 3))
    hit5 = float(np.mean(rankings <= 5))

    return mrr, mr, hit1, hit3, hit5


def train_local(model_args, data,client_id,global_rel=False):
    global args, model, sess
    args = model_args
    client_id = client_id
    #print(client_id)
    # extract data
    triplets,nentity,n_relations,fed_data = data
    print(n_relations)

    _, valid_triplets, test_triplets = triplets

    #print(test_triplets)

    train_triplets = fed_data[client_id-1]

    train_triples = np.stack(train_triplets)
    #train_triples = np.stack(train_triplets[:int(len(train_triplets)*0.8)])
    #print(train_triples)

    valid_triples = np.stack(valid_triplets)

    test_triples = np.stack(test_triplets)
    #test_triples = np.stack(train_triplets[int(len(train_triplets)*0.8):])


    all_triples = np.concatenate([train_triples, valid_triples, test_triples])
    train_dataset = TrainDataset(train_triples, nentity, args.num_neg)
    #valid_dataset = TestDataset(valid_triples, all_triples, nentity)
    #test_dataset = TestDataset(test_triples, all_triples, nentity)
    test_dataset = TestDataset(test_triples, all_triples, nentity)
    testrel_dataset = TestRelDataset(test_triples, all_triples, n_relations)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.local_batch_size,
        shuffle=True,
        collate_fn=TrainDataset.collate_fn
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        collate_fn=TestDataset.collate_fn
    )
    testrel_dataloader = DataLoader(
        testrel_dataset,
        batch_size=args.test_batch_size,
        collate_fn=TestRelDataset.collate_fn
    )


    #标签为r
    # define the model
    
    relbert = np.load('../data/' + args.dataset + '/kgemodel/' + 'relbert' +str(client_id)+ '.npy')
            #print(bert.shape)
    if global_rel == False:
        rel_embed = torch.tensor(relbert).cuda().requires_grad_()
    else:
        global_rel = np.load('../data/' + args.dataset + '/relbertfed.npy')
        global_rel = torch.tensor(global_rel[:-1])
        relbert = torch.tensor(relbert)
        rel_embed = ((relbert+global_rel)/2).cuda().requires_grad_()
        #rel_embed = torch.cat((relbert,global_rel),1).cuda().requires_grad_()
    #rel_embed = torch.tensor(relbert).cuda().requires_grad_()
    #rel_embed = torch.zeros(n_relations, args.hidden_dim).cuda().requires_grad_()
    entbert = np.load('../data/' + args.dataset + '/kgemodel/' + 'entitybert' + str(client_id)+'.npy')

    ent_embed = torch.tensor(entbert).cuda().requires_grad_()
    #ent_embed = torch.zeros(nentity, args.hidden_dim).cuda().requires_grad_()


    embedding_range = torch.Tensor([(args.gamma + args.epsilon) / args.hidden_dim])
    # nn.init.uniform_(
    #         tensor=ent_embed,
    #         a=-embedding_range.item(),
    #         b=embedding_range.item()
    #     )

    # nn.init.uniform_(
    #         tensor=rel_embed,
    #         a=-embedding_range.item(),
    #         b=embedding_range.item()
    #     )
                    

    kge_model = KGEModel(args, args.model).cuda()
    kgerel_model = KGERelModel(args,args.model).cuda()

  
    

    optimizer = optim.Adam([{'params': rel_embed},
                                {'params': ent_embed}], lr=args.local_lr)

    losses = []
    for i in range(args.local_epoch):
        for batch in train_dataloader:
            positive_sample, negative_sample, sample_idx = batch

            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()

            negative_score = kge_model((positive_sample, negative_sample),
                                                 rel_embed, ent_embed)

            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
                                  * F.logsigmoid(-negative_score)).sum(dim=1)

            positive_score = kge_model(positive_sample,
                                                rel_embed, ent_embed, neg=False)

            positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()

            loss = (positive_sample_loss + negative_sample_loss) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        #eval tail
        # results = ddict(float)
        # for batch in test_dataloader:
        #     triplets, labels = batch
        #     #print(labels.shape) 16 *14541
        #     #labels
        #     triplets, labels = triplets.cuda(), labels.cuda()
        #     head_idx, rel_idx, tail_idx = triplets[:, 0], triplets[:, 1], triplets[:, 2]
        #     #print(tail_idx)
        #     #print('error occur')
        #     pred = kge_model((triplets, None),
        #                            rel_embed, ent_embed) #batch_size*节点数量
        #     b_range = torch.arange(pred.size()[0]).cuda()
        #     #print(pred.size()[0])
        #     #pred.size()[0] 16 因为test数据集的batch size是16
        #     #print(tail_idx.shape)#16
        #     target_pred = pred[b_range, tail_idx]
        #     #print(target_pred.shape)
        #     #print(target_pred) 16
        #     #print('lable')
        #     #print(labels.byte())

        #     #torch.where(a>0,a,b)      #合并a,b两个tensor a>0的地方保存
        #     pred = torch.where(labels.byte(), -torch.ones_like(pred) * 10000000, pred)
        #     pred[b_range, tail_idx] = target_pred
        #     #print(pred[0].tolist())
        #     #break

        #     #对分数进行排序 然后选取实际节点的预测分数
        #     ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True),
        #                               dim=1, descending=False)[b_range, tail_idx]

        #     #print(ranks)
        #     ranks = ranks.float()
        #     #torch.numel  显示矩阵的所有元素个数
        #     count = torch.numel(ranks)
        #     #print(count)

        #     results['count'] += count
        #     results['mr'] += torch.sum(ranks).item()
        #     results['mrr'] += torch.sum(1.0 / ranks).item()

        #     for k in [1,3, 5, 10]:
        #         results['hits@{}'.format(k)] += torch.numel(ranks[ranks <= k])

        # for k, v in results.items():
        #     if k != 'count':
        #         results[k] /= results['count']
        # #print('test')
        # print(results)

    


        #eval rel
    
        results = ddict(float)
        for batch in testrel_dataloader:
            triplets, labels = batch
                #print(labels.shape) 16 *14541
                #labels
            triplets, labels = triplets.cuda(), labels.cuda()
            head_idx, rel_idx, tail_idx = triplets[:, 0], triplets[:, 1], triplets[:, 2]
                #print(tail_idx)
                #print('error occur')
            pred = kgerel_model((triplets, None),
                                    rel_embed, ent_embed) #batch_size*节点数量
                #b_range = torch.arange(pred.size()[0], device=args.gpu)
            b_range = torch.arange(pred.size()[0]).cuda()
                #print(pred.size()[0])
                #pred.size()[0] 16 因为test数据集的batch size是16
                #print(tail_idx.shape)#16
            target_pred = pred[b_range, rel_idx]
  
                #对分数进行排序 然后选取实际节点的预测分数
            ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True),
                                        dim=1, descending=False)[b_range, rel_idx]

                #print(ranks)
            ranks = ranks.float()
                #torch.numel  显示矩阵的所有元素个数
            count = torch.numel(ranks)
                #print(count)

            results['count'] += count
            results['mr'] += torch.sum(ranks).item()
            results['mrr'] += torch.sum(1.0 / ranks).item()

            for k in [1, 3,5, 10]:
                results['hits@{}'.format(k)] += torch.numel(ranks[ranks <= k])

        for k, v in results.items():
            if k != 'count':
                results[k] /= results['count']

        print(results)

    rel_embed = rel_embed.data.cpu().numpy()
    print(rel_embed.shape)
    np.save('/home/common/lzh17/PathCon/PathCon-master/data/'+ args.dataset+'/feddata/relbert'+str(client_id)+'.npy',rel_embed)

    ent_embed = ent_embed.data.cpu().numpy()
    np.save('/home/common/lzh17/PathCon/PathCon-master/data/'+ args.dataset+'/feddata/entbert'+str(client_id)+'.npy',ent_embed)

def train_local_entrel(model_args, data,global_rel=False):
    global args, model, sess
    args = model_args
    #client_id = client_id
    #print(client_id)
    # extract data
    triplets,nentity,n_relations,fed_data = data
    print(n_relations)

    train_triplets, valid_triplets, test_triplets = triplets

    #print(test_triplets)

    #train_triplets = fed_data[client_id-1]


    train_triples = np.stack(train_triplets)
    #print(train_triples)

    valid_triples = np.stack(valid_triplets)

    test_triples = np.stack(test_triplets)


    all_triples = np.concatenate([train_triples, valid_triples, test_triples])
    train_dataset = TrainDataset(train_triples, nentity, args.num_neg)
    #valid_dataset = TestDataset(valid_triples, all_triples, nentity)
    #test_dataset = TestDataset(test_triples, all_triples, nentity)
    test_dataset = TestDataset(test_triples, all_triples, nentity)
    testrel_dataset = TestRelDataset(test_triples, all_triples, n_relations)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.local_batch_size,
        shuffle=True,
        collate_fn=TrainDataset.collate_fn
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        collate_fn=TestDataset.collate_fn
    )
    testrel_dataloader = DataLoader(
        testrel_dataset,
        batch_size=args.test_batch_size,
        collate_fn=TestRelDataset.collate_fn
    )


    #标签为r
    # define the model
    
    relbert = np.load('../data/' + args.dataset + '/relbertnew.npy')
            #print(bert.shape)
    if global_rel == False:
        rel_embed = torch.tensor(relbert).cuda().requires_grad_()
    else:
        global_rel = np.load('../data/' + args.dataset + '/relbertfed.npy')
        global_rel = torch.tensor(global_rel[:-1])
        relbert = torch.tensor(relbert)
        rel_embed = torch.cat((relbert,global_rel),1).cuda().requires_grad_()

    #rel_embed = torch.zeros(n_relations, args.hidden_dim).cuda().requires_grad_()
    entbert = np.load('../data/' + args.dataset + '/entitybertnew.npy')

    ent_embed = torch.tensor(entbert).cuda().requires_grad_()
    #ent_embed = torch.zeros(nentity, args.hidden_dim).cuda().requires_grad_()


    embedding_range = torch.Tensor([(args.gamma + args.epsilon) / args.hidden_dim])
    # nn.init.uniform_(
    #         tensor=ent_embed,
    #         a=-embedding_range.item(),
    #         b=embedding_range.item()
    #     )

    # nn.init.uniform_(
    #         tensor=rel_embed,
    #         a=-embedding_range.item(),
    #         b=embedding_range.item()
    #     )
                    

    kge_model = KGEModel(args, args.model).cuda()
    kgerel_model = KGERelModel(args,args.model)

  
    

    optimizer = optim.Adam([{'params': rel_embed},
                                {'params': ent_embed}], lr=args.local_lr)

    losses = []
    for i in range(args.local_epoch):
        for batch in train_dataloader:
            positive_sample, negative_sample, sample_idx = batch

            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            print(rel_embed)
            negative_score = kge_model((positive_sample, negative_sample),
                                                 rel_embed, ent_embed)

            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
                                  * F.logsigmoid(-negative_score)).sum(dim=1)

            positive_score = kge_model(positive_sample,
                                                rel_embed, ent_embed, neg=False)

            positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()

            loss = (positive_sample_loss + negative_sample_loss) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        #eval tail
        results = ddict(float)
        for batch in test_dataloader:
            triplets, labels = batch
            #print(labels.shape) 16 *14541
            #labels
            triplets, labels = triplets.cuda(), labels.cuda()
            head_idx, rel_idx, tail_idx = triplets[:, 0], triplets[:, 1], triplets[:, 2]
            #print(tail_idx)
            #print('error occur')
            pred = kge_model((triplets, None),
                                   rel_embed, ent_embed) #batch_size*节点数量
            b_range = torch.arange(pred.size()[0]).cuda()
            #print(pred.size()[0])
            #pred.size()[0] 16 因为test数据集的batch size是16
            #print(tail_idx.shape)#16
            target_pred = pred[b_range, tail_idx]
            #print(target_pred.shape)
            #print(target_pred) 16
            #print('lable')
            #print(labels.byte())

            #torch.where(a>0,a,b)      #合并a,b两个tensor a>0的地方保存
            pred = torch.where(labels.byte(), -torch.ones_like(pred) * 10000000, pred)
            pred[b_range, tail_idx] = target_pred
            #print(pred[0].tolist())
            #break

            #对分数进行排序 然后选取实际节点的预测分数
            ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True),
                                      dim=1, descending=False)[b_range, tail_idx]

            #print(ranks)
            ranks = ranks.float()
            #torch.numel  显示矩阵的所有元素个数
            count = torch.numel(ranks)
            #print(count)

            results['count'] += count
            results['mr'] += torch.sum(ranks).item()
            results['mrr'] += torch.sum(1.0 / ranks).item()

            for k in [1,3, 5, 10]:
                results['hits@{}'.format(k)] += torch.numel(ranks[ranks <= k])

        for k, v in results.items():
            if k != 'count':
                results[k] /= results['count']
        #print('test')
        print(results)

    rel_embed = rel_embed.data.cpu().numpy()
    print(rel_embed.shape)
    np.save('/home/common/lzh17/PathCon/PathCon-master/data/'+ args.dataset+'/relbertnew2.npy',rel_embed)

    ent_embed = ent_embed.data.cpu().numpy()
    np.save('/home/common/lzh17/PathCon/PathCon-master/data/'+ args.dataset+'/entitybertnew2.npy',ent_embed)


        #eval rel
        # if i %10 ==0:
        #     results = ddict(float)
        #     for batch in testrel_dataloader:
        #         triplets, labels = batch
        #         #print(labels.shape) 16 *14541
        #         #labels
        #         triplets, labels = triplets.cuda(), labels.cuda()
        #         head_idx, rel_idx, tail_idx = triplets[:, 0], triplets[:, 1], triplets[:, 2]
        #         #print(tail_idx)
        #         #print('error occur')
        #         pred = kge_model((triplets, None),
        #                             rel_embed, ent_embed) #batch_size*节点数量
        #         #b_range = torch.arange(pred.size()[0], device=args.gpu)
        #         b_range = torch.arange(pred.size()[0]).cuda()
        #         #print(pred.size()[0])
        #         #pred.size()[0] 16 因为test数据集的batch size是16
        #         #print(tail_idx.shape)#16
        #         target_pred = pred[b_range, tail_idx]
  
        #         #对分数进行排序 然后选取实际节点的预测分数
        #         ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True),
        #                                 dim=1, descending=False)[b_range, tail_idx]

        #         #print(ranks)
        #         ranks = ranks.float()
        #         #torch.numel  显示矩阵的所有元素个数
        #         count = torch.numel(ranks)
        #         #print(count)

        #         results['count'] += count
        #         results['mr'] += torch.sum(ranks).item()
        #         results['mrr'] += torch.sum(1.0 / ranks).item()

        #         for k in [1, 3,5, 10]:
        #             results['hits@{}'.format(k)] += torch.numel(ranks[ranks <= k])

        #     for k, v in results.items():
        #         if k != 'count':
        #             results[k] /= results['count']

        #     print(results)

def test_global(model_args, data,state_dict=None):
     
    global args, model, sess
    args = model_args

    # extract data
    #triplets, paths, n_relations, path_params,fed_data,valid_test_data = data
    triplets, paths, n_relations, neighbor_params, path_params = data

    train_triplets, valid_triplets, test_triplets = triplets

    valid_entity_pairs = torch.LongTensor(np.array([[triplet[0], triplet[1]] for triplet in valid_triplets], np.int32))
    test_entity_pairs = torch.LongTensor(np.array([[triplet[0], triplet[1]] for triplet in test_triplets], np.int32))


    #neighbor_params = fed_data[2][client_id]
    #print(type(train_paths))   
    # valid_entity_pairs = valid_test_data[0]
    # test_entity_pairs = valid_test_data[2]

    _, valid_paths, test_paths = paths

    #标签为r
    
    # valid_labels = valid_test_data[1]
    # test_labels = valid_test_data[3]
    valid_labels = torch.LongTensor(np.array([triplet[2] for triplet in valid_triplets], np.int32))
    test_labels = torch.LongTensor(np.array([triplet[2] for triplet in test_triplets], np.int32))

    # define the model
    
    model = PathCon(args, n_relations, neighbor_params, path_params)
    if state_dict != None:
        model.load_state_dict(state_dict)
    #print(model.state_dict())
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.lr,
        # weight_decay=args.l2,
    )

    if args.cuda:
        model = model.cuda()

        valid_labels = valid_labels.cuda()
        test_labels = test_labels.cuda()
        if args.use_context:
            valid_entity_pairs = valid_entity_pairs.cuda()
            test_entity_pairs = test_entity_pairs.cuda()

    # prepare for top-k evaluation
    #集合为值的字典
    #true_relations (head,tail)可能存在的relation
    true_relations = defaultdict(set)
    for head, tail, relation in train_triplets + valid_triplets + test_triplets:
        true_relations[(head, tail)].add(relation)
    best_valid_acc = 0.0
    final_res = None  # acc, mrr, mr, hit1, hit3, hit5

    #print('round '+str(round)+': client '+str(client_id)+' start training ...'+'dataset: '+args.dataset)

    test_acc, test_scores = evaluate(test_entity_pairs, test_paths, test_labels)

        # show evaluation result for current epoch
    current_res = 'acc: %.4f' % test_acc
        #print('train acc: %.4f   valid acc: %.4f   test acc: %.4f' % (train_acc, valid_acc, test_acc))
    mrr, mr, hit1, hit3, hit5 = calculate_ranking_metrics(test_triplets, test_scores, true_relations)
    current_res += '   mrr: %.4f   mr: %.4f   h1: %.4f   h3: %.4f   h5: %.4f' % (mrr, mr, hit1, hit3, hit5)
        #print('           mrr: %.4f   mr: %.4f   h1: %.4f   h3: %.4f   h5: %.4f' % (mrr, mr, hit1, hit3, hit5))
    print()



    # show final evaluation result
    print(' final results\n%s' % current_res)
    
