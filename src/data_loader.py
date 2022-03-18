import os
import re
import pickle
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from utils import count_all_paths_with_mp, count_paths, get_path_dict_and_length, one_hot_path_id, sample_paths,one_hot_path_id_single
import torch
import random

# defaultdict 使用dict时，如果引用的Key不存在，就会抛出KeyError。如果希望key不存在时，返回一个默认值，就可以用defaultdict
# 字典的元素是集合 
entity2edge_set = defaultdict(set)  # entity id -> set of (both incoming and outgoing) edges connecting to this entity
entity2edges = []  # each row in entity2edges is the sampled edges connecting to this entity
edge2entities = []  # each row in edge2entities is the two entities connected by this edge
edge2relation = []  # each row in edge2relation is the relation type of this edge


#relation path 与实体连接的 (r,e)对集合
e2re = defaultdict(set)  # entity index -> set of pair (relation, entity) connecting to this entity


def read_entities(file_name):
    d = {}
    file = open(file_name)
    for line in file:
        index, name = line.strip().split('\t')
        d[name] = int(index)
    file.close()

    return d


def read_relations(file_name):
    bow = []
    count_vec = CountVectorizer()

    d = {}
    file = open(file_name)
    for line in file:
        index, name = line.strip().split('\t')
        d[name] = int(index)

        if args.feature_type == 'bow' and not os.path.exists('../data/' + args.dataset + '/bow.npy'):
            tokens = re.findall('[a-z]{2,}', name)
            bow.append(' '.join(tokens))
    file.close()

    if args.feature_type == 'bow' and not os.path.exists('../data/' + args.dataset + '/bow.npy'):
        bow = count_vec.fit_transform(bow)
        np.save('../data/' + args.dataset + '/bow.npy', bow.toarray())

    return d


def read_triplets(file_name):
    data = []

    file = open(file_name)
    for line in file:
        head, relation, tail = line.strip().split('\t')

        head_idx = entity_dict[head]
        relation_idx = relation_dict[relation]
        tail_idx = entity_dict[tail]

        data.append((head_idx, tail_idx, relation_idx))
    file.close()

    return data

def read_hrtriplets(file_name):
    data = []

    file = open(file_name)
    for line in file:
        head, relation, tail = line.strip().split('\t')

        head_idx = entity_dict[head]
        relation_idx = relation_dict[relation]
        tail_idx = entity_dict[tail]

        data.append((head_idx, relation_idx,tail_idx))
    file.close()

    return data

def build_kg(train_data):
    for edge_idx, triplet in enumerate(train_data):
        head_idx, tail_idx, relation_idx = triplet

        if args.use_context:
            entity2edge_set[head_idx].add(edge_idx)
            entity2edge_set[tail_idx].add(edge_idx)
            edge2entities.append([head_idx, tail_idx])
            edge2relation.append(relation_idx)

        if args.use_path:
            e2re[head_idx].add((relation_idx, tail_idx))
            e2re[tail_idx].add((relation_idx, head_idx))

    # To handle the case where a node does not appear in the training data (i.e., this node has no neighbor edge),
    # we introduce a null entity (ID: n_entities), a null edge (ID: n_edges), and a null relation (ID: n_relations).
    # entity2edge_set[isolated_node] = {null_edge}
    # entity2edge_set[null_entity] = {null_edge}
    # edge2entities[null_edge] = [null_entity, null_entity]
    # edge2relation[null_edge] = null_relation
    # The feature of null_relation is a zero vector. See _build_model() of model.py for details

    #对一个没出现在训练集的节点进行处理，节点id 为已有节点id+1 关系同理
    if args.use_context:
        null_entity = len(entity_dict)
        null_relation = len(relation_dict)
        null_edge = len(edge2entities)
        edge2entities.append([null_entity, null_entity])
        edge2relation.append(null_relation)

        for i in range(len(entity_dict) + 1):
            if i not in entity2edge_set:
                entity2edge_set[i] = {null_edge}
            sampled_neighbors = np.random.choice(list(entity2edge_set[i]), size=args.neighbor_samples,
                                                 replace=len(entity2edge_set[i]) < args.neighbor_samples)
            entity2edges.append(sampled_neighbors)


def build_kg_fed(train_data):

    entity2edge_set = defaultdict(set)  # entity id -> set of (both incoming and outgoing) edges connecting to this entity
    entity2edges = []  # each row in entity2edges is the sampled edges connecting to this entity
    edge2entities = []  # each row in edge2entities is the two entities connected by this edge
    edge2relation = []  # each row in edge2relation is the relation type of this edge


    #relation path 与实体连接的 (r,e)对集合
    e2re = defaultdict(set)  # entity index -> set of pair (relation, entity) connecting to this entity

    for edge_idx, triplet in enumerate(train_data):
        head_idx, tail_idx, relation_idx = triplet

        if args.use_context:
            entity2edge_set[head_idx].add(edge_idx)
            entity2edge_set[tail_idx].add(edge_idx)
            edge2entities.append([head_idx, tail_idx])
            edge2relation.append(relation_idx)

        if args.use_path:
            e2re[head_idx].add((relation_idx, tail_idx))
            e2re[tail_idx].add((relation_idx, head_idx))

    # To handle the case where a node does not appear in the training data (i.e., this node has no neighbor edge),
    # we introduce a null entity (ID: n_entities), a null edge (ID: n_edges), and a null relation (ID: n_relations).
    # entity2edge_set[isolated_node] = {null_edge}
    # entity2edge_set[null_entity] = {null_edge}
    # edge2entities[null_edge] = [null_entity, null_entity]
    # edge2relation[null_edge] = null_relation
    # The feature of null_relation is a zero vector. See _build_model() of model.py for details

    #对一个没出现在训练集的节点进行处理，节点id 为已有节点id+1 关系同理
    if args.use_context:
        null_entity = len(entity_dict)
        null_relation = len(relation_dict)
        null_edge = len(edge2entities)
        edge2entities.append([null_entity, null_entity])
        edge2relation.append(null_relation)

        for i in range(len(entity_dict) + 1):
            if i not in entity2edge_set:
                entity2edge_set[i] = {null_edge}
            sampled_neighbors = np.random.choice(list(entity2edge_set[i]), size=args.neighbor_samples,
                                                 replace=len(entity2edge_set[i]) < args.neighbor_samples)
            entity2edges.append(sampled_neighbors)

    return entity2edges,edge2entities,edge2relation,e2re


def get_h2t(train_triplets, valid_triplets, test_triplets):

    # 返回(h,t)对的字典
    head2tails = defaultdict(set)
    for head, tail, relation in train_triplets + valid_triplets + test_triplets:
        head2tails[head].add(tail)
    return head2tails

def get_h2t_single(triplets):
     # 返回(h,t)对的字典
    head2tails = defaultdict(set)
    for head, tail, relation in triplets:
        head2tails[head].add(tail)
    return head2tails


def get_path(triplets,path_name,e2re):
    directory = '../data/' + args.dataset + '/cache/'
    length = str(args.max_path_len)

    if not os.path.exists(directory):
        os.mkdir(directory)

    if os.path.exists(directory + path_name+'_' + length + '.pkl'):
        # 路径文件已经存在
        print('loading paths from files ...')
        paths = pickle.load(open(directory + path_name+'_' + length + '.pkl', 'rb'))
        

    else:
        #生成路径文件
        print('counting paths from head to tail ...')
        #get_h2t 返回头尾节点对的字典 字典元素是h的t实体集合
        head2tails = get_h2t_single(triplets)
        #print(head2tails)
        
        ht2paths = count_all_paths_with_mp(e2re, args.max_path_len, [(k, v) for k, v in head2tails.items()])
        #ht2paths ?
        train_set = set(triplets)
        paths = count_paths(triplets, ht2paths, train_set)
        
        print('dumping paths to files ...')
        pickle.dump(paths, open(directory + path_name+'_' + length + '.pkl', 'wb'))
        

    #返回的类型是列表
    return paths

def get_paths(train_triplets, valid_triplets, test_triplets):
    directory = '../data/' + args.dataset + '/cache/'
    length = str(args.max_path_len)
    
    if not os.path.exists(directory):
        os.mkdir(directory)

    if os.path.exists(directory + 'train_paths_' + length + '.pkl'):
        # 路径文件已经存在
        print('loading paths from files ...')
        train_paths = pickle.load(open(directory + 'train_paths_' + length + '.pkl', 'rb'))
        valid_paths = pickle.load(open(directory + 'valid_paths_' + length + '.pkl', 'rb'))
        test_paths = pickle.load(open(directory + 'test_paths_' + length + '.pkl', 'rb'))
        #print(train_paths)

    else:
        #生成路径文件
        print('counting paths from head to tail ...')
        #get_h2t 返回头尾节点对的字典 字典元素是h的t实体集合
        head2tails = get_h2t(train_triplets, valid_triplets, test_triplets)
        #print(head2tails)
        #print(head2tails)
        #print(e2re)
        ht2paths = count_all_paths_with_mp(e2re, args.max_path_len, [(k, v) for k, v in head2tails.items()])
        #print(ht2paths)
        #ht2paths h t 对中包含的路径
        train_set = set(train_triplets)
        train_paths = count_paths(train_triplets, ht2paths, train_set)
        valid_paths = count_paths(valid_triplets, ht2paths, train_set)
        test_paths = count_paths(test_triplets, ht2paths, train_set)

        #print(train_paths)
        print('dumping paths to files ...')
        pickle.dump(train_paths, open(directory + 'train_paths_' + length + '.pkl', 'wb'))
        pickle.dump(valid_paths, open(directory + 'valid_paths_' + length + '.pkl', 'wb'))
        pickle.dump(test_paths, open(directory + 'test_paths_' + length + '.pkl', 'wb'))

    # if using rnn and no path is found for the triplet, put an empty path into paths
    if args.path_type == 'rnn':
        for paths in train_paths + valid_paths + test_paths:
            if len(paths) == 0:
                paths.append([])

    #返回的类型是列表
    return train_paths, valid_paths, test_paths


def load_data(model_args):
    global args, entity_dict, relation_dict
    args = model_args
    directory = '../data/' + args.dataset + '/'

    print('reading entity dict and relation dict ...')
    entity_dict = read_entities(directory + 'entities.dict')
    relation_dict = read_relations(directory + 'relations.dict')

    print('reading train, validation, and test data ...')
    #read_triplets 返回的是（hid,tid,rid）的列表
    train_triplets = read_triplets(directory + 'train.txt')
    valid_triplets = read_triplets(directory + 'valid.txt')
    test_triplets = read_triplets(directory + 'test.txt')

    


    print('processing the knowledge graph ...')
    build_kg(train_triplets)
    '''
    print(e2re[2])
    all_paths = [[i] for i in e2re[2]]
    print(all_paths)
    path = all_paths[0]
    print(path)
    print(path[-1][1])
    print(e2re[12632])
    for edge in e2re[12632]:
        print(edge[1])
        print(path+[edge])
        all_paths.append(path+[edge])
        print(all_paths)
        break
    '''
    triplets = [train_triplets, valid_triplets, test_triplets]
    #print(triplets)
    print(len(entity_dict))#40943
    print(len(entity2edges))#40944
    print(len(edge2entities))#86836
    print(len(edge2relation))#86836
    if args.use_context:

        #entity2edges 节点连着边（随机取的边）
        #edge2entities 边连着的头尾节点 h,t 
        #edge2relation 边的类型
        neighbor_params = [np.array(entity2edges), np.array(edge2entities), np.array(edge2relation)]
    else:
        neighbor_params = None

    if args.use_path:
        train_paths, valid_paths, test_paths = get_paths(train_triplets, valid_triplets, test_triplets)
        print(len(train_paths))
        print(len(valid_paths))
        print(len(test_paths))
        path2id, id2path, id2length = get_path_dict_and_length(
            train_paths, valid_paths, test_paths, len(relation_dict), args.max_path_len)
        
        #print(len(path2id))#1751 1751条路径

        if args.path_type == 'embedding':
            print('transforming paths to one hot IDs ...')
            #独热编码向量
            paths = one_hot_path_id(train_paths, valid_paths, test_paths, path2id)#[train 86835*1751 valid 3034*1751 test 3134*1751 ]
            #print(paths.shape)
            path_params = [len(path2id)]
            print(path_params)
        elif args.path_type == 'rnn':
            paths = sample_paths(train_paths, valid_paths, test_paths, path2id, args.path_samples)
            path_params = [id2path, id2length]
        else:
            raise ValueError('unknown path type')
    else:
        paths = [None] * 3
        path_params = None


    return triplets, paths, len(relation_dict), neighbor_params, path_params



def load_data_fed(model_args):
    global args, entity_dict, relation_dict
    args = model_args
    directory = '../data/' + args.dataset + '/'

    print('reading entity dict and relation dict ...')
    entity_dict = read_entities(directory + 'entities.dict')
    relation_dict = read_relations(directory + 'relations.dict')

    print('reading train, validation, and test data ...')
    #read_triplets 返回的是（hid,tid,rid）的列表
    train_triplets = read_triplets(directory + 'trainshuffle.txt')
    valid_triplets = read_triplets(directory + 'valid.txt')
    test_triplets = read_triplets(directory + 'test.txt')
    
    build_kg(train_triplets)
    #if os.path.exists(directory + 'feddata/' + 'train1.txt'):

    random.seed(10)
    # random.shuffle(train_triplets)

    triplets = [train_triplets, valid_triplets, test_triplets]
    if args.use_path:
        train_paths, valid_paths, test_paths = get_paths(train_triplets, valid_triplets, test_triplets)
        path2id, id2path, id2length = get_path_dict_and_length(
            train_paths, valid_paths, test_paths, len(relation_dict), args.max_path_len)
        paths = one_hot_path_id(train_paths, valid_paths, test_paths, path2id)    

    client_num = args.client_num

    #return 0
    #print(valid_paths)

    triplets_fed = []
    kg_fed = []  
    neighbor_fed = []
    path_fed = []
    path_params = [len(path2id)]
    train_edges_fed = []
    train_entity_pairs_fed = []
    train_labels_fed = []

    for i in range(client_num):
        #triplets_fed.append(train_triplets[i*int(len(train_triplets)/client_num):(i+1)*int(len(train_triplets)/client_num)])

        #unbanlance
        if args.unbalance == True:
            index = [0,0.4,0.7,0.85,0.95,1]
            triplets_fed.append(train_triplets[int(index[i]*len(train_triplets)):int(index[i+1]*len(train_triplets))])
        else:
            triplets_fed.append(train_triplets[i*int(len(train_triplets)/client_num):(i+1)*int(len(train_triplets)/client_num)])

        kg_fed.append(build_kg_fed(triplets_fed[i]))
        if args.use_context:
            neighbor_params = [np.array(kg_fed[i][0]), np.array(kg_fed[i][1]), np.array(kg_fed[i][2])]
            neighbor_fed.append(neighbor_params)
        else:
            neighbor_params = None

        if args.path_type == 'embedding':
            
            #独热编码向量
            if args.unbalance == True:
                train_path = get_path(triplets_fed[i],'train_path_ub'+str(i),kg_fed[i][3])
            else:
                train_path = get_path(triplets_fed[i],'train_path'+str(i),kg_fed[i][3])
            #print(train_paths)
            print('transforming paths to one hot IDs ...')
            path = one_hot_path_id_single(train_path,path2id)
            #print(path.shape)
            path_fed.append(path)

        train_edges = torch.LongTensor(np.array(range(len(triplets_fed[i])), np.int32))
        train_edges_fed.append(train_edges)

        train_entity_pairs = torch.LongTensor(np.array([[triplet[0], triplet[1]] for triplet in triplets_fed[i]], np.int32))
        train_entity_pairs_fed.append(train_entity_pairs)
        
        train_labels = torch.LongTensor(np.array([triplet[2] for triplet in triplets_fed[i]], np.int32))
        train_labels_fed.append(train_labels)
   
        


    valid_entity_pairs = torch.LongTensor(np.array([[triplet[0], triplet[1]] for triplet in valid_triplets], np.int32))
    test_entity_pairs = torch.LongTensor(np.array([[triplet[0], triplet[1]] for triplet in test_triplets], np.int32))

    valid_labels = torch.LongTensor(np.array([triplet[2] for triplet in valid_triplets], np.int32))
    test_labels = torch.LongTensor(np.array([triplet[2] for triplet in test_triplets], np.int32))

    fed_data = triplets_fed,kg_fed,neighbor_fed,path_fed,train_edges_fed,train_entity_pairs_fed,train_labels_fed

    valid_test_data = valid_entity_pairs,valid_labels,test_entity_pairs,test_labels
            

    return triplets, paths, len(relation_dict), path_params,fed_data,valid_test_data



def load_data_fed_local(model_args):
    global args, entity_dict, relation_dict
    args = model_args
    directory = '../data/' + args.dataset + '/'

    print('reading entity dict and relation dict ...')
    entity_dict = read_entities(directory + 'entities.dict')
    relation_dict = read_relations(directory + 'relations.dict')

    print('reading train, validation, and test data ...')
    #read_triplets 返回的是（hid,tid,rid）的列表
    train_triplets = read_hrtriplets(directory + 'trainshuffle.txt')
    valid_triplets = read_hrtriplets(directory + 'valid.txt')
    test_triplets = read_hrtriplets(directory + 'test.txt')
    
    build_kg(train_triplets)
    #if os.path.exists(directory + 'feddata/' + 'train1.txt'):

    random.seed(10)
    # random.shuffle(train_triplets)

    triplets = [train_triplets, valid_triplets, test_triplets]
  

    client_num = args.client_num

    #return 0
    #print(valid_paths)

    triplets_fed = []
    kg_fed = []  


    for i in range(client_num):
        triplets_fed.append(train_triplets[i*int(len(train_triplets)/client_num):(i+1)*int(len(train_triplets)/client_num)])


    fed_data = triplets_fed
    
            

    return triplets,len(entity_dict),len(relation_dict), fed_data