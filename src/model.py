import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from aggregators import MeanAggregator, ConcatAggregator, CrossAggregator

class PathCon(nn.Module):
    def __init__(self, args, n_relations, params_for_neighbors, params_for_paths,id = 1):
        #n_relation 关系数量
        super(PathCon, self).__init__()
        self._parse_args(args, n_relations, params_for_neighbors, params_for_paths,id)
        self._build_model()

    def _parse_args(self, args, n_relations, params_for_neighbors, params_for_paths,id):
        self.n_relations = n_relations
        self.use_gpu = args.cuda

        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.hidden_dim = args.dim
        self.feature_type = args.feature_type

        self.use_context = args.use_context
        self.dataset = args.dataset
        self.id = id

        #使用节点嵌入
        self.use_entemb = args.use_entemb
        self.use_localrel = args.use_localrel

        if self.use_localrel:
            bert = np.load('../data/' + self.dataset + '/' + 'relbertnew.npy')
            self.local_rel = torch.tensor(bert).cuda()

        if self.use_entemb:
            if self.dataset == 'DDB14':
                random_seed = 123
                torch.manual_seed(random_seed)
                self.entity_dim = 768
                self.entity_features =  torch.randn(9203, 768).cuda()
                self.entity_features = torch.cat([self.entity_features, 
                            torch.zeros([1, self.entity_dim]).cuda() if self.use_gpu \
                                else torch.zeros([1, self.entity_dim])], dim=0)
                self.entity_features.requires_grad = False
            else:
                print('use entemb')
                bert = np.load('../data/' + self.dataset + '/' + 'entitybertnew.npy')
                #print(self.id)
                #bert = np.load('../data/' + self.dataset + '/feddata/' + 'entbert'+str(self.id)+'.npy')
                print(bert.shape)
                self.entity_dim = bert.shape[1]
                self.entity_features = torch.tensor(bert).cuda() if self.use_gpu \
                        else torch.tensor(bert).requires_grad_()
                self.entity_features = torch.cat([self.entity_features, 
                            torch.zeros([1, self.entity_dim]).cuda() if self.use_gpu \
                                else torch.zeros([1, self.entity_dim])], dim=0)

                #self.entity_features = nn.Parameter(self.entity_features)
        
        if self.use_context:
            self.entity2edges = torch.LongTensor(params_for_neighbors[0]).cuda() if args.cuda \
                    else torch.LongTensor(params_for_neighbors[0])
            self.edge2entities = torch.LongTensor(params_for_neighbors[1]).cuda() if args.cuda \
                    else torch.LongTensor(params_for_neighbors[1])
            self.edge2relation = torch.LongTensor(params_for_neighbors[2]).cuda() if args.cuda \
                    else torch.LongTensor(params_for_neighbors[2])
            self.neighbor_samples = args.neighbor_samples
            self.context_hops = args.context_hops
            if args.neighbor_agg == 'mean':
                self.neighbor_agg = MeanAggregator
            elif args.neighbor_agg == 'concat':
                self.neighbor_agg = ConcatAggregator
            elif args.neighbor_agg == 'cross':
                self.neighbor_agg = CrossAggregator

        self.use_path = args.use_path
        if self.use_path:
            self.path_type = args.path_type
            if self.path_type == 'embedding':
                self.n_paths = params_for_paths[0]
            elif self.path_type == 'rnn':
                self.max_path_len = args.max_path_len
                self.path_samples = args.path_samples
                self.path_agg = args.path_agg
                self.id2path = torch.LongTensor(params_for_paths[0]).cuda() if args.cuda \
                        else torch.LongTensor(params_for_paths[0])
                self.id2length = torch.LongTensor(params_for_paths[1]).cuda() if args.cuda \
                        else torch.LongTensor(params_for_paths[1])

    def _build_model(self):
        # define initial relation features
        #_build_relation_feature 构建关系特征 self.relation_dim self.relation_features
        if self.use_context or (self.use_path and self.path_type == 'rnn'):
            self._build_relation_feature()

        self.scores = 0.0

        if self.use_context:
            self.aggregators = nn.ModuleList(self._get_neighbor_aggregators())  # define aggregators for each layer

        if self.use_path:
            if self.path_type == 'embedding':
                self.layer = nn.Linear(self.n_paths, self.n_relations)
                nn.init.xavier_uniform_(self.layer.weight)

            elif self.path_type == 'rnn':
                self.rnn = nn.LSTM(input_size=self.relation_dim, hidden_size=self.hidden_dim, batch_first=True)
                self.layer = nn.Linear(self.hidden_dim, self.n_relations)
                nn.init.xavier_uniform_(self.layer.weight)

    def forward(self, batch):
        if self.use_context:
            self.entity_pairs = batch['entity_pairs']
            self.train_edges = batch['train_edges']

        if self.use_path:
            if self.path_type == 'embedding':
                self.path_features = batch['path_features']
            elif self.path_type == 'rnn':
                self.path_ids = batch['path_ids']

        self.labels = batch['labels']

        self._call_model()

    def _call_model(self):
        self.scores = 0.
        #print(self.scores.shape)

        if self.use_context:
            edge_list, entity_list,mask_list = self._get_neighbors_and_masks(self.labels, self.entity_pairs, self.train_edges)
            #print(len(edge_list))
            self.aggregated_neighbors = self._aggregate_neighbors(edge_list,entity_list, mask_list)
            self.scores += self.aggregated_neighbors

        if self.use_path:
            if self.path_type == 'embedding':
                #print("path shape")
                #print(self.path_features.shape)
                self.scores += self.layer(self.path_features)

            elif self.path_type == 'rnn':
                rnn_output = self._rnn(self.path_ids)
                self.scores += self._aggregate_paths(rnn_output)
        #print(self.scores.shape)

        self.scores_normalized = F.sigmoid(self.scores)
    #构建关系特征
    def _build_relation_feature(self):
        if self.feature_type == 'id':
            self.relation_dim = self.n_relations
            self.relation_features = torch.eye(self.n_relations).cuda() if self.use_gpu \
                    else torch.eye(self.n_relations)
        elif self.feature_type == 'bow':
            bow = np.load('../data/' + self.dataset + '/bow.npy')
            self.relation_dim = bow.shape[1]
            self.relation_features = torch.tensor(bow).cuda() if self.use_gpu \
                    else torch.tensor(bow)
        elif self.feature_type == 'bert':
            #bert = np.load('../data/' + self.dataset + '/' + self.feature_type + '.npy')
            bert = np.load('../data/' + self.dataset + '/relbert.npy')
            #print(bert.shape)
            self.relation_dim = bert.shape[1]
            
            self.relation_features = torch.tensor(bert).cuda() if self.use_gpu \
                    else torch.tensor(bert).requires_grad_()
        elif self.feature_type == 'random':
            bert = np.load('../data/' + self.dataset + '/relbert.npy')
            self.relation_dim = bert.shape[1]
            self.relation_features = torch.zeros(self.n_relations, self.relation_dim).cuda().requires_grad_()

        # the feature of the last relation (the null relation) is a zero vector
        #添加不存在关系特征向量
        # if self.use_localrel == True:
        #     self.relation_features = (self.relation_features+self.local_rel)/2
        
        self.relation_features = torch.cat([self.relation_features, 
                        torch.zeros([1, self.relation_dim]).cuda() if self.use_gpu \
                            else torch.zeros([1, self.relation_dim])], dim=0)
        if self.use_entemb:
                self.relation_dim = self.relation_dim+self.entity_dim
        self.relation_features = nn.Parameter(self.relation_features)

    # 从点获取邻居边，从边获取连接节点 （轮流）
    #获取掩码 不在训练集的掩盖
    # edge_list c
    def _get_neighbors_and_masks(self, relations, entity_pairs, train_edges):
        #relations = labels
        edges_list = [relations]
        entity_list = []
        masks = []
        #train_edges 是边的编号
        train_edges = torch.unsqueeze(train_edges, -1)  # [batch_size, 1]
        #print(train_edges)

        for i in range(self.context_hops):
            if i == 0:
                neighbor_entities = entity_pairs
                #print(neighbor_entities.view(-1))
            else:

                #torch.index_select(input, dim, index, out=None) 函数返回的是沿着输入张量的指定维度的指定索引号进行索引的张量子集，其中输入张量、指定维度和指定索引号就是
                #edge2entities 是边的节点对 的列表
                #edge2entities = []  # each row in edge2entities is the two entities connected by this edge
                neighbor_entities = torch.index_select(self.edge2entities, 0, 
                            edges_list[-1].view(-1)).view([self.batch_size, -1])
            #entity2edges = []  # each row in entity2edges is the sampled edges connecting to this entity
            #print(self.entity2edges[0])
            #print(neighbor_entities.view(-1).shape)
            neighbor_edges = torch.index_select(self.entity2edges, 0, 
                            neighbor_entities.view(-1)).view([self.batch_size, -1])
            
            edges_list.append(neighbor_edges)
            entity_list.append(neighbor_entities.view(-1).view(self.batch_size,-1))
            

            #掩码 遮盖 训练的边
            mask = neighbor_edges - train_edges  # [batch_size, -1]
            mask = (mask != 0).float()
            #print(mask)
            masks.append(mask)
        #print(len(entity_list))
        # for l in edges_list:
        #     print(l.shape)
        # for e in entity_list:
        #     print(e.shape)
        return edges_list, entity_list,masks


    #构建聚集器
    def _get_neighbor_aggregators(self):
        aggregators = []  # store all aggregators

        if self.context_hops == 1:
            aggregators.append(self.neighbor_agg(batch_size=self.batch_size,
                                                 input_dim=self.relation_dim,
                                                 output_dim=self.n_relations,
                                                 self_included=False))
        else:
            # the first layer
            aggregators.append(self.neighbor_agg(batch_size=self.batch_size,
                                                 input_dim=self.relation_dim,
                                                 output_dim=self.hidden_dim,
                                                 act=F.relu))
            # middle layers
            for i in range(self.context_hops - 2):
                aggregators.append(self.neighbor_agg(batch_size=self.batch_size,
                                                     input_dim=self.hidden_dim,
                                                     output_dim=self.hidden_dim,
                                                     act=F.relu))
            # the last layer
            aggregators.append(self.neighbor_agg(batch_size=self.batch_size,
                                                 input_dim=self.hidden_dim,
                                                 output_dim=self.n_relations,
                                                 self_included=False))
        return aggregators

    #聚集邻居信息
    def _aggregate_neighbors(self, edge_list, entity_list,mask_list):
        # translate edges IDs to relations IDs, then to features
        #edge_list[0] = [labels]
        entity_vectors=[]
        
                #print(entity_vectors[0].shape)
                #break

        edge_vectors = [torch.index_select(self.relation_features, 0, edge_list[0])]
        for edges in edge_list[1:]:
            relations = torch.index_select(self.edge2relation, 0, 
                            edges.view(-1)).view(list(edges.shape)+[-1])
            #edge_vectors 存储边的关系向量
            edge_vectors.append(torch.index_select(self.relation_features, 0, 
                            relations.view(-1)).view(list(relations.shape)+[-1]))

        if self.use_entemb:
            
            for entities in entity_list:
                #print(entities)
                #break
                entities = entities.view(list(entities.shape)+[-1])

                entity_vectors.append(torch.index_select(self.entity_features, 0, 
                            entities.view(-1)).view(list(entities.shape)+[-1]))


            for i in range(len(entity_vectors)):
                
                temp_edge_vectors = edge_vectors[i+1].view([self.batch_size,entity_vectors[i].shape[1],-1,edge_vectors[i+1].shape[-1]])

                #add 方法
                # edge_vectors[i+1] = (temp_edge_vectors+entity_vectors[i]).view(list(edge_vectors[i+1].shape))
                #print(edge_vectors[i+1].shape)
                
                #concat 方法
                temp = torch.zeros(temp_edge_vectors.shape).cuda()
                temp = temp + entity_vectors[i]

                edge_vectors[i+1] = torch.cat((temp_edge_vectors,temp),-1).view(list(edge_vectors[i+1].shape)[:-1]+[-1])
            #concat 方法
            edge_vectors[0] = torch.cat((edge_vectors[0],edge_vectors[0]),-1)

        # shape of edge vectors:
        # [[batch_size, relation_dim],
        # 
        #  [batch_size, 2 * neighbor_samples, relation_dim],   2 代表 实体对两个实体
        #  [batch_size, (2 * neighbor_samples) ^ 2, relation_dim],
        #  ...]
        #print(len(edge_vectors))#3
        # for edge in edge_vectors:
        #     print(edge.shape)
        # for entity in entity_vectors:
        #     print(entity.shape)
        for i in range(self.context_hops):
            aggregator = self.aggregators[i]
            edge_vectors_next_iter = []
            neighbors_shape = [self.batch_size, -1, 2, self.neighbor_samples, aggregator.input_dim]
           
            masks_shape = [self.batch_size, -1, 2, self.neighbor_samples, 1]

            for hop in range(self.context_hops - i):
                # print("sss")
                # print(edge_vectors[hop].shape)
                # print(edge_vectors[hop+1].shape)
                # print(edge_vectors[hop+1].view(neighbors_shape).shape)
                # print(entity_vectors[hop].shape)
                # print("sss")
                vector = aggregator(self_vectors=edge_vectors[hop],
                                    neighbor_vectors=edge_vectors[hop + 1].view(neighbors_shape),
                                    masks=mask_list[hop].view(masks_shape))
                edge_vectors_next_iter.append(vector)
            edge_vectors = edge_vectors_next_iter

        # edge_vectos[0]: [self.batch_size, 1, self.n_relations]
        res = edge_vectors[0].view([self.batch_size, self.n_relations])
        return res

    def _rnn(self, path_ids):
        path_ids = path_ids.view([self.batch_size * self.path_samples])  # [batch_size * path_samples]
        paths = torch.index_select(self.id2path, 0, 
                path_ids.view(-1)).view(list(path_ids.shape)+[-1])  # [batch_size * path_samples, max_path_len]
        # [batch_size * path_samples, max_path_len, relation_dim]
        path_features = torch.index_select(self.relation_features, 0, 
                paths.view(-1)).view(list(paths.shape)+[-1])
        lengths = torch.index_select(self.id2length, 0, path_ids)  # [batch_size * path_samples]

        output, _ = self.rnn(path_features)
        output = torch.cat([torch.zeros(output.shape[0], 1, output.shape[2]).cuda() if self.use_gpu \
                    else torch.zeros(output.shape[0], 1, output.shape[2]), output], dim=1)
        output = output.gather(1, lengths.unsqueeze(-1).unsqueeze(-1).expand(output.shape[0], 1, output.shape[-1]))

        output = self.layer(output)
        output = output.view([self.batch_size, self.path_samples, self.n_relations])

        return output

    def _aggregate_paths(self, inputs):
        # input shape: [batch_size, path_samples, n_relations]

        if self.path_agg == 'mean':
            output = torch.mean(inputs, dim=1)  # [batch_size, n_relations]
        elif self.path_agg == 'att':
            assert self.use_context
            aggregated_neighbors = self.aggregated_neighbors.unsqueeze(1)  # [batch_size, 1, n_relations]
            attention_weights = torch.sum(aggregated_neighbors * inputs, dim=-1)  # [batch_size, path_samples]
            attention_weights = F.softmax(attention_weights, dim=-1)  # [batch_size, path_samples]
            attention_weights = attention_weights.unsqueeze(-1)  # [batch_size, path_samples, 1]
            output = torch.sum(attention_weights * inputs, dim=1)  # [batch_size, n_relations]
        else:
            raise ValueError('unknown path_agg')

        return output

    @staticmethod
    def train_step(model, optimizer, batch):
        model.train()
        optimizer.zero_grad()
        model(batch)
        criterion = nn.CrossEntropyLoss()
        loss = torch.mean(criterion(model.scores, model.labels))
        
        loss.backward()
        optimizer.step()

        return loss.item()
    
    @staticmethod
    def test_step(model, batch):
        model.eval()
        with torch.no_grad():
            model(batch)
            acc = (model.labels == model.scores.argmax(dim=1)).float().tolist()
        return acc, model.scores_normalized.tolist()