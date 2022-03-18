import argparse
from data_loader import load_data,load_data_fed,load_data_fed_local
from train import train,train_fed,train_local,train_local_entrel,test_global
import os
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np


def print_setting(args):
    assert args.use_context or args.use_path
    print()
    print('=============================================')
    print('dataset: ' + args.dataset)
    print('epoch: ' + str(args.epoch))
    print('batch_size: ' + str(args.batch_size))
    print('dim: ' + str(args.dim))
    print('l2: ' + str(args.l2))
    print('lr: ' + str(args.lr))
    print('feature_type: ' + args.feature_type)

    print('use relational context: ' + str(args.use_context))
    if args.use_context:
        print('context_hops: ' + str(args.context_hops))
        print('neighbor_samples: ' + str(args.neighbor_samples))
        print('neighbor_agg: ' + args.neighbor_agg)

    print('use relational path: ' + str(args.use_path))
    if args.use_path:
        print('max_path_len: ' + str(args.max_path_len))
        print('path_type: ' + args.path_type)
        if args.path_type == 'rnn':
            print('path_samples: ' + str(args.path_samples))
            print('path_agg: ' + args.path_agg)
    print('=============================================')
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=True, help='use gpu', action='store_true')

    '''
    # ===== FB15k ===== #
    parser.add_argument('--dataset', type=str, default='FB15k', help='dataset name')
    parser.add_argument('--epoch', type=int, default=20, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--dim', type=int, default=64, help='hidden dimension')
    parser.add_argument('--l2', type=float, default=1e-7, help='l2 regularization weight')
    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--feature_type', type=str, default='id', help='type of relation features: id, bow, bert')

    # settings for relational context
    parser.add_argument('--use_context', type=bool, default=True, help='whether use relational context')
    parser.add_argument('--context_hops', type=int, default=2, help='number of context hops')
    parser.add_argument('--neighbor_samples', type=int, default=32, help='number of sampled neighbors for one hop')
    parser.add_argument('--neighbor_agg', type=str, default='concat', help='neighbor aggregator: mean, concat, cross')

    # settings for relational path
    parser.add_argument('--use_path', type=bool, default=True, help='whether use relational path')
    parser.add_argument('--max_path_len', type=int, default=2, help='max length of a path')
    parser.add_argument('--path_type', type=str, default='embedding', help='path representation type: embedding, rnn')
    parser.add_argument('--path_samples', type=int, default=8, help='number of sampled paths if using rnn')
    parser.add_argument('--path_agg', type=str, default='att', help='path aggregator if using rnn: mean, att')
    '''

    
    # ===== FB15k-237 ===== #
    # parser.add_argument('--dataset', type=str, default='FB15k-237', help='dataset name')
    # parser.add_argument('--epoch', type=int, default=20, help='number of epochs')
    # parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    # parser.add_argument('--dim', type=int, default=64, help='hidden dimension')
    # parser.add_argument('--l2', type=float, default=1e-7, help='l2 regularization weight')
    # parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    # parser.add_argument('--feature_type', type=str, default='bert', help='type of relation features: id, bow, bert')


    # parser.add_argument('--use_entemb', type=bool, default=True, help='whether use relational entemb')
    # # settings for relational context
    # parser.add_argument('--use_context', type=bool, default=True, help='whether use relational context')
    # parser.add_argument('--context_hops', type=int, default=2, help='number of context hops')
    # parser.add_argument('--neighbor_samples', type=int, default=8, help='number of sampled neighbors for one hop')
    # parser.add_argument('--neighbor_agg', type=str, default='concat', help='neighbor aggregator: mean, concat, cross')

    # # settings for relational path
    # parser.add_argument('--use_path', type=bool, default=True, help='whether use relational path')
    # parser.add_argument('--max_path_len', type=int, default=2, help='max length of a path')
    # parser.add_argument('--path_type', type=str, default='embedding', help='path representation type: embedding, rnn')
    # parser.add_argument('--path_samples', type=int, default=8, help='number of sampled paths if using rnn')
    # parser.add_argument('--path_agg', type=str, default='att', help='path aggregator if using rnn: mean, att')
    #     #setting for fed
    # parser.add_argument('--client_num', type=int, default=5, help='client_num')
    # parser.add_argument('--fed_epoch', type=int, default=1, help='client_num')
    
    
    '''
    # ===== wn18 ===== #
    parser.add_argument('--dataset', type=str, default='wn18', help='dataset name')
    parser.add_argument('--epoch', type=int, default=20, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--dim', type=int, default=64, help='hidden dimension')
    parser.add_argument('--l2', type=float, default=1e-7, help='l2 regularization weight')
    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--feature_type', type=str, default='id', help='type of relation features: id, bow, bert')

    # settings for relational context
    parser.add_argument('--use_context', type=bool, default=True, help='whether use relational context')
    parser.add_argument('--context_hops', type=int, default=3, help='number of context hops')
    parser.add_argument('--neighbor_samples', type=int, default=16, help='number of sampled neighbors for one hop')
    parser.add_argument('--neighbor_agg', type=str, default='cross', help='neighbor aggregator: mean, concat, cross')

    # settings for relational path
    parser.add_argument('--use_path', type=bool, default=True, help='whether use relational path')
    parser.add_argument('--max_path_len', type=int, default=3, help='max length of a path')
    parser.add_argument('--path_type', type=str, default='embedding', help='path representation type: embedding, rnn')
    parser.add_argument('--path_samples', type=int, default=8, help='number of sampled paths if using rnn')
    parser.add_argument('--path_agg', type=str, default='att', help='path aggregator if using rnn: mean, att')
    '''
    '''
    # ===== wn18rr ===== #
    parser.add_argument('--dataset', type=str, default='wn18rr', help='dataset name')
    parser.add_argument('--epoch', type=int, default=20, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--dim', type=int, default=64, help='hidden dimension')
    parser.add_argument('--l2', type=float, default=1e-7, help='l2 regularization weight')
    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--feature_type', type=str, default='id', help='type of relation features: id, bow, bert')

    # settings for relational context
    parser.add_argument('--use_context', type=bool, default=True, help='whether use relational context')
    parser.add_argument('--context_hops', type=int, default=3, help='number of context hops')
    parser.add_argument('--neighbor_samples', type=int, default=8, help='number of sampled neighbors for one hop')
    parser.add_argument('--neighbor_agg', type=str, default='cross', help='neighbor aggregator: mean, concat, cross')

    # settings for relational path
    parser.add_argument('--use_path', type=bool, default=True, help='whether use relational path')
    parser.add_argument('--max_path_len', type=int, default=4, help='max length of a path')
    parser.add_argument('--path_type', type=str, default='embedding', help='path representation type: embedding, rnn')
    parser.add_argument('--path_samples', type=int, default=8, help='number of sampled paths if using rnn')
    parser.add_argument('--path_agg', type=str, default='att', help='path aggregator if using rnn: mean, att')
    #setting for fed
    parser.add_argument('--client_num', type=int, default=5, help='client_num')
    parser.add_argument('--fed_epoch', type=int, default=1, help='client_num')

    '''

    
    # ===== NELL995 ===== #
    #train local epoch 10
    #train fed epoch 11
    parser.add_argument('--dataset', type=str, default='NELL995', help='dataset name')
    parser.add_argument('--epoch', type=int, default=20, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--dim', type=int, default=64, help='hidden dimension')
    parser.add_argument('--l2', type=float, default=1e-7, help='l2 regularization weight')
    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--feature_type', type=str, default='random', help='type of relation features: id, bow, bert,random')

    parser.add_argument('--use_entemb', type=bool, default=False, help='whether use entemb')
    parser.add_argument('--use_localrel', type=bool, default=False, help='whether use relational emb')
    # settings for relational context
    parser.add_argument('--use_context', type=bool, default=True, help='whether use relational context')
    parser.add_argument('--context_hops', type=int, default=2, help='number of context hops')
    parser.add_argument('--neighbor_samples', type=int, default=8, help='number of sampled neighbors for one hop')
    parser.add_argument('--neighbor_agg', type=str, default='concat', help='neighbor aggregator: mean, concat, cross')

    # settings for relational path
    parser.add_argument('--use_path', type=bool, default=True, help='whether use relational path')
    parser.add_argument('--max_path_len', type=int, default=3, help='max length of a path')
    parser.add_argument('--path_type', type=str, default='embedding', help='path representation type: embedding, rnn')
    parser.add_argument('--path_samples', type=int, default=8, help='number of sampled paths if using rnn')
    parser.add_argument('--path_agg', type=str, default='att', help='path aggregator if using rnn: mean, att')
        #setting for fed
    parser.add_argument('--client_num', type=int, default=5, help='client_num')
    parser.add_argument('--fed_epoch', type=int, default=1, help='client_num')
    

    
    # ===== DDB14 ===== #
    # parser.add_argument('--dataset', type=str, default='DDB14', help='dataset name')
    # parser.add_argument('--epoch', type=int, default=20, help='number of epochs')
    # parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    # parser.add_argument('--dim', type=int, default=64, help='hidden dimension')
    # parser.add_argument('--l2', type=float, default=1e-7, help='l2 regularization weight')
    # parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    # parser.add_argument('--feature_type', type=str, default='bert', help='type of relation features: id, bow, bert')


    # parser.add_argument('--use_entemb', type=bool, default=False, help='whether use relational entemb')
    # # settings for relational context
    # parser.add_argument('--use_context', type=bool, default=True, help='whether use relational context')
    # parser.add_argument('--context_hops', type=int, default=3, help='number of context hops')
    # parser.add_argument('--neighbor_samples', type=int, default=4, help='number of sampled neighbors for one hop')
    # parser.add_argument('--neighbor_agg', type=str, default='concat', help='neighbor aggregator: mean, concat, cross')

    # # settings for relational path
    # parser.add_argument('--use_path', type=bool, default=True, help='whether use relational path')
    # parser.add_argument('--max_path_len', type=int, default=3, help='max length of a path')
    # parser.add_argument('--path_type', type=str, default='embedding', help='path representation type: embedding, rnn')
    # parser.add_argument('--path_samples', type=int, default=8, help='number of sampled paths if using rnn')
    # parser.add_argument('--path_agg', type=str, default='att', help='path aggregator if using rnn: mean, att')
    #     #setting for fed
    # parser.add_argument('--client_num', type=int, default=5, help='client_num')
    # parser.add_argument('--fed_epoch', type=int, default=1, help='client_num')
    

    

    parser.add_argument('--model', default='TransE', choices=['TransE', 'RotatE', 'DistMult','ComplEx'])

    # one task hyperparam



    parser.add_argument('--local_batch_size', default=128, type=int)
    parser.add_argument('--test_batch_size', default=16, type=int)
    parser.add_argument('--num_neg', default=128, type=int)
    parser.add_argument('--local_lr', default=0.001, type=int)

    # for local training
    parser.add_argument('--local_epoch', default=10)
    parser.add_argument('--fraction', default=1, type=float)
    
    parser.add_argument('--gamma', default=10.0, type=float)
    parser.add_argument('--epsilon', default=2.0, type=float)
    parser.add_argument('--hidden_dim', default=128, type=float)
    parser.add_argument('--gpu', default='1', type=str)
    parser.add_argument('--num_cpu', default=10, type=int)
    parser.add_argument('--adversarial_temperature', default=1.0, type=float)

    # parser.add_argument('--negative_adversarial_sampling', default=True, type=bool)
    parser.add_argument('--seed', default=12345, type=int)
    parser.add_argument('--unbalance', type=bool, default=True, help='unbalance data')

    #args.gpu = torch.device('cuda:' + args.gpu)


    # loacl setting


    args = parser.parse_args()
    print_setting(args)
    #data = load_data(args)
    #train(args, data)


    #train fed
    data = load_data_fed(args)
    local_state = {}

    
    # balance

    if args.unbalance == False:
        for epoch in range(args.epoch):
            print(str(epoch)+' round')
            if epoch == 21 and args.use_localrel == True:
                bert = np.load('../data/' + args.dataset + '/' + 'relbertnew.npy')
                local_rel = torch.tensor(bert)
                local_rel = torch.cat([local_rel, torch.zeros([1, bert.shape[1]])]).cuda()
                global_state['relation_features'] = (global_state['relation_features']+local_rel)/2
                
            
            for id in range(args.client_num):
                if epoch == 0:
                    local_state[id] = train_fed(args,data,epoch,id)
                else:
                    local_state[id] = train_fed(args,data,epoch,id,global_state)

            if epoch == 0:
                global_state = local_state[0].copy()

            #全局清零   
            for s in global_state:
                global_state[s] = global_state[s]-global_state[s]

            # 更新           
            for id in range(args.client_num):

                for s in global_state:
                    global_state[s] += local_state[id][s]*(1/args.client_num)


        global_data = load_data(args)
        if  args.use_localrel == True:
                bert = np.load('../data/' + args.dataset + '/' + 'relbertnew.npy')
                local_rel = torch.tensor(bert)
                local_rel = torch.cat([local_rel, torch.zeros([1, bert.shape[1]])]).cuda()
                global_state['relation_features'] = (global_state['relation_features']+local_rel)/2
        test_global(args,global_data,global_state)
        rel_embed = global_state['relation_features'].data.cpu().numpy()
        #np.save('/home/common/lzh17/PathCon/PathCon-master/data/'+ args.dataset+'/relbertfed.npy',rel_embed)

    #unblance
    else:
        for epoch in range(args.epoch):
            print(str(epoch)+' round')
            
            for id in range(args.client_num):
                if epoch == 0:
                    local_state[id] = train_fed(args,data,epoch,id)
                else:

                    for  s in global_state:
                        if s =='entity_features':
                            continue
                        local_state[id][s] = global_state[s]
                    local_state[id] = train_fed(args,data,epoch,id,local_state[id])

            if epoch == 0:
                global_state = local_state[0].copy()

            #全局清零   
            for s in global_state:
                global_state[s] = global_state[s]-global_state[s]

            # 更新 
            p = [0.4,0.3,0.15,0.1,0.05]          
            for id in range(args.client_num):

                for s in global_state:
                    global_state[s] += local_state[id][s]*p[id]
    
    ######
    # for epoch in range(args.epoch):
    #     print(str(epoch)+' round')
        
    #     for id in range(args.client_num):
    #         if epoch == 0:
    #             local_state[id] = train_fed(args,data,epoch,id)
    #         else:
    #             local_state[id]['relation_features'] = global_state['relation_features']
    #             local_state[id] = train_fed(args,data,epoch,id,local_state[id])

    #     if epoch == 0:
    #         global_state = local_state[0].copy()

    #     #全局清零   
    #     # for s in global_state:
    #     #     global_state[s] = global_state[s]-global_state[s]
    #     global_state['relation_features'] = global_state['relation_features']-global_state['relation_features']


    #     # 更新           
    #     for id in range(args.client_num):

            
    #         global_state['relation_features'] += local_state[id]['relation_features']*(1/args.client_num)
    

    # train local

    # data = load_data_fed_local(args)
    # # #train_local_entrel(args,data,True)
    # for i in range(1,6):
    #     print(i)
    #     train_local(args,data,i,False)
    #     break

    
    # args.fed_epoch = 20
    # train_fed(args,data,0,0)
    # # train_fed(args,data,0,1)
    # # train_fed(args,data,0,2)
    # # train_fed(args,data,0,3)
    # # train_fed(args,data,0,4)

    


    


if __name__ == '__main__':
    main()
