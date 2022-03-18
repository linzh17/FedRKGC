import numpy as np
import multiprocessing as mp
import scipy.sparse as sp
from collections import defaultdict


def count_all_paths_with_mp(e2re, max_path_len, head2tails):
    #head2tails 头尾实体对 列表
    #e2re  与实体连接的 (r,e)对集合
    #print(e2re)
    n_cores, pool, range_list = get_params_for_mp(len(head2tails))
    #range_list 为各个核处理的数据索引
    results = pool.map(count_all_paths, zip([e2re] * n_cores,
                                            [max_path_len] * n_cores,
                                            [head2tails[i[0]:i[1]] for i in range_list],
                                            range(n_cores)))
    res = defaultdict(set)
    #update 方法 将一个字典加入到字典中
    for ht2paths in results:
        res.update(ht2paths)

    return res


def get_params_for_mp(n_triples):
    n_cores = mp.cpu_count()
    pool = mp.Pool(n_cores)
    avg = n_triples // n_cores

    range_list = []
    start = 0
    for i in range(n_cores):
        num = avg + 1 if i < n_triples - avg * n_cores else avg
        range_list.append([start, start + num])
        start += num

    return n_cores, pool, range_list


# input: [(h1, {t1, t2 ...}), (h2, {t3 ...}), ...]
# output: {(h1, t1): paths, (h1, t2): paths, (h2, t3): paths, ...}
def count_all_paths(inputs):
    e2re, max_path_len, head2tails, pid = inputs
    ht2paths = {}
    for i, (head, tails) in enumerate(head2tails):
        ht2paths.update(bfs(head, tails, e2re, max_path_len))
        print('pid %d:  %d / %d' % (pid, i, len(head2tails)))
    print('pid %d  done' % pid)
    return ht2paths


def bfs(head, tails, e2re, max_path_len):
    # put length-1 paths into all_paths
    # each element in all_paths is a path consisting of a sequence of (relation, entity)
    all_paths = [[i] for i in e2re[head]]

    p = 0
    for length in range(2, max_path_len + 1):
        while p < len(all_paths) and len(all_paths[p]) < length:
            path = all_paths[p]
            #取路径最后一个实体 e.g((r1,t1),(r4,t4))
            last_entity_in_path = path[-1][1]
            #路径上的实体 e.g head ,t1,t4
            entities_in_path = set([head] + [i[1] for i in path])
            #将最后的实体下一跳路径中前面路径未出现过的实体添加如open list
            for edge in e2re[last_entity_in_path]:
                # append (relation, entity) to the path if the new entity does not appear in this path before
                if edge[1] not in entities_in_path:
                    all_paths.append(path + [edge])
            p += 1

    ht2paths = defaultdict(set)
    for path in all_paths:
        tail = path[-1][1]
        if tail in tails:  # if this path ends at tail
            ht2paths[(head, tail)].add(tuple([i[0] for i in path]))

    return ht2paths


def count_paths(triplets, ht2paths, train_set):
    #train_set 训练集三元组的集合
    res = []

    for head, tail, relation in triplets:
        path_set = ht2paths[(head, tail)]
        if (tail, head, relation) in train_set:
            path_list = list(path_set)
        else:
            path_list = list(path_set - {tuple([relation])})
        res.append([list(i) for i in path_list])

    return res


def get_path_dict_and_length(train_paths, valid_paths, test_paths, null_relation, max_path_len):
    path2id = {}
    id2path = []
    id2length = []
    n_paths = 0

    for paths_of_triplet in train_paths + valid_paths + test_paths:
        for path in paths_of_triplet:
            #print(path)
            #print(tuple(path))
            path_tuple = tuple(path)
            if path_tuple not in path2id:
                path2id[path_tuple] = n_paths
                id2length.append(len(path))
                #print([null_relation])
                id2path.append(path + [null_relation] * (max_path_len - len(path)))  # padding
                n_paths += 1
            #break
    #print(path2id)
    return path2id, id2path, id2length


def one_hot_path_id(train_paths, valid_paths, test_paths, path_dict):
    res = []
    
    for data in (train_paths, valid_paths, test_paths):
        bop_list = []  # bag of paths
        for paths in data:
            bop_list.append([path_dict[tuple(path)] for path in paths])
        res.append(bop_list)

    return [get_sparse_feature_matrix(bop_list, len(path_dict)) for bop_list in res]

def one_hot_path_id_single(train_paths, path_dict):
    res = []

    bop_list = []  # bag of paths
    
    for paths in train_paths:
        bop_list.append([path_dict[tuple(path)] for path in paths])


    return get_sparse_feature_matrix(bop_list, len(path_dict)) 

def sample_paths(train_paths, valid_paths, test_paths, path_dict, path_samples):
    res = []
    for data in [train_paths, valid_paths, test_paths]:
        path_ids_for_data = []
        for paths in data:
            path_ids_for_triplet = [path_dict[tuple(path)] for path in paths]
            sampled_path_ids_for_triplet = np.random.choice(
                path_ids_for_triplet, size=path_samples, replace=len(path_ids_for_triplet) < path_samples)
            path_ids_for_data.append(sampled_path_ids_for_triplet)

        path_ids_for_data = np.array(path_ids_for_data, dtype=np.int32)
        res.append(path_ids_for_data)
    return res


def get_sparse_feature_matrix(non_zeros, n_cols):
    #lil_matrix使用两个列表保存非零元素。data保存每行中的非零元素，rows保存非零元素所在的列。这种格式也很适合逐个添加元素，并且能快速获取行相关的数据。
    features = sp.lil_matrix((len(non_zeros), n_cols), dtype=np.float64)
    for i in range(len(non_zeros)):
        for j in non_zeros[i]:
            features[i, j] = +1.0
    #print(features)
    return features


def sparse_to_tuple(sparse_matrix):
    if not sp.isspmatrix_coo(sparse_matrix):
        sparse_matrix = sparse_matrix.tocoo()
    indices = np.vstack((sparse_matrix.row, sparse_matrix.col)).transpose()
    values = sparse_matrix.data
    shape = sparse_matrix.shape
    #coo_matrix采用三个数组row、col和data保存非零元素的信息。这三个数组的长度相同，row保存元素的行，col保存元素的列，data保存元素的值。
    #print(indices)
    #print(indices)
    # print(len(values))
    # print(shape)
    return indices, values, shape
