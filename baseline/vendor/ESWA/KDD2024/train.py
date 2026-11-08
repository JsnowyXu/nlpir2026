# -*- coding:utf-8 -*-
import json
import sys
import os
from pathlib import Path

from torch.nn.utils.rnn import pad_sequence

from models.dhpt import DHPT

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
path_project = os.path.split(root_path)[0]
sys.path.append(root_path)
sys.path.append(path_project)
sys.path.append('/')

import time
import datetime
import logging
import torch
import numpy as np
import pickle
from tqdm import tqdm

import gc

from utils.data_helpers import load_adjacent
gc.collect()

from evals.evaluation import evaluation
from utils.metric import get_metric
from utils.util import EarlyStopMonitor, set_random_seed, save_results
from modules.history_memory import HistoryMemory

sys.path.append('/')

from utils import data_helpers as dh
from utils import param_parser as parser
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
args = parser.parameter_parser()


Path(cur_path + f"/saved_checkpoints/{args.US}").mkdir(parents=True, exist_ok=True)
Path(cur_path + f"/results/{args.US}/").mkdir(parents=True, exist_ok=True)
Path(cur_path + f"/logs/{args.US}/").mkdir(parents=True, exist_ok=True)
Path(cur_path + f"/memories/{args.US}/").mkdir(parents=True, exist_ok=True)
Path(cur_path + f"/modellog/{args.US}/").mkdir(parents=True, exist_ok=True)

model_type = 'word' + str(args.level)
if args.use_bert:
    model_type = 'bert'+ str(args.level)
if args.use_cnn:
    model_type += '_cnn'
if args.use_company:
    model_type += '_company'+str(args.weight)
if args.use_hierarchy:
    model_type += '_hier'
if args.use_memory:
    model_type += '_memory'
    if args.use_his_memory:
        model_type += '_his'
    if args.use_time:
        model_type += "_time"
if args.use_his_memory:
    Path(cur_path + f"/IPC/{args.US}/{model_type}/").mkdir(parents=True, exist_ok=True)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(cur_path +'/logs/{}/{}-{}.log'.format(args.US,model_type, str(time.time())))

fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

get_checkpoint_path = lambda \
        epoch:cur_path + f'/saved_checkpoints/{args.US}/{model_type}-{epoch}.pth'

set_random_seed(2024)

if __name__ == '__main__':
    startTime = datetime.datetime.now()
    logger.info("startTime:{}".format(startTime))
    start_epoch = time.time()
    
    dh.tab_printer(args, logger)
    early_stopper = EarlyStopMonitor(max_round=args.patience)
    
    # Load word2vec model from train
    word2idx, embedding_matrix = dh.load_word2vec_matrix(args.word2vec_file.format(args.US))
  
    # Load sentences, labels, and training parameters
    GPU = args.gpu
    device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_string)

    # 层级关系矩阵，8*124，124*609； total_number:各层级数量列表[8,124,609]
    hierarchy_data, total_number = dh.load_hierarchy_tree(args.hierarchy_tree, args.US, args.hierarchy_names)
    logger.info("classification codes in hierarchical tree is:{}".format(total_number))
      
    raw_node_features = np.load(args.raw_node_features.format(args.US)) 
    # load train,valid,test datasets
   
    train_data, company_map, companies, patent_map, patents, company_dt = \
        dh.load_data_and_labels(args, args.train_file.format(args.US, args.US), word2idx,level= args.level)
    print("end construct train data")
    # pklTrainDic={}
    # pklTrainDic['train_data'] = train_data
    # pklTrainDic['company_map'] = company_map
    # pklTrainDic['companies'] = companies
    # pklTrainDic['patent_map'] = patent_map
    # pklTrainDic['patents'] = patents
    # pklTrainDic['company_dt'] = company_dt

    # with open(cur_path +'/train'+str(args.level) + args.US + str(args.max_len_bert)+'.pkl', 'wb') as file:
    #     pickle.dump(pklTrainDic, file)

    # with open(cur_path +'/train'+str(args.level) + args.US + str(args.max_len_bert)+'.pkl', 'rb') as file:
    #     # 加载数据到变量中
    #     train = pickle.load(file) 
    #     train_data = train['train_data']
    #     company_map= train['company_map']
    #     companies= train['companies']
    #     patent_map= train['patent_map']
    #     patents= train['patents']
    #     company_dt= train['company_dt']
    # print("end construct train data: " + str(patents))

    val_data, company_map, companies, patent_map, patents, company_dt = \
        dh.load_data_and_labels(args, args.validation_file.format(args.US, args.US), word2idx,level= args.level,
                                company_maps=company_map, company_ids=companies,
                                patent_maps=patent_map, patent_ids=patents, company_dts=company_dt, type='val')
    # pklValDic={}
    # pklValDic['val_data'] = val_data
    # pklValDic['company_map'] = company_map
    # pklValDic['companies'] = companies
    # pklValDic['patent_map'] = patent_map
    # pklValDic['patents'] = patents
    # pklValDic['company_dt'] = company_dt

    # with open(cur_path + '/val'+str(args.level) + args.US  + str(args.max_len_bert)+'.pkl', 'wb') as file:
    #     pickle.dump(pklValDic, file)
    
    # with open(cur_path +'/val'+str(args.level) + args.US + str(args.max_len_bert)+'.pkl', 'rb') as file:
    #     # 加载数据到变量中
    #     val = pickle.load(file) 
    #     val_data = val['val_data']
    #     company_map= val['company_map']
    #     companies= val['companies']
    #     patent_map= val['patent_map']
    #     patents= val['patents']
    #     company_dt= val['company_dt']
    # print("end construct val data: " + str(len(val_data["patentId"])))

    test_data, company_map, companies, patent_map, patents, company_dt = \
        dh.load_data_and_labels(args, args.test_file.format(args.US, args.US), word2idx,level= args.level,
                                company_maps=company_map, company_ids=companies,
                                patent_maps=patent_map, patent_ids=patents, company_dts=company_dt, type='test')
    # pklTestDic={}
    # pklTestDic['test_data'] = test_data
    # pklTestDic['company_map'] = company_map
    # pklTestDic['companies'] = companies
    # pklTestDic['patent_map'] = patent_map
    # pklTestDic['patents'] = patents
    # pklTestDic['company_dt'] = company_dt

    # with open(cur_path + '/test'+str(args.level) + args.US  + str(args.max_len_bert)+'.pkl', 'wb') as file:
    #     pickle.dump(pklTestDic, file)
    
    # with open(cur_path +'/test'+str(args.level) + args.US + str(args.max_len_bert) +'.pkl', 'rb') as file:
    #     # 加载数据到变量中
    #     test = pickle.load(file) 
    #     test_data = test['test_data']
    #     company_map= test['company_map']
    #     companies= test['companies']
    #     patent_map= test['patent_map']
    #     patents= test['patents']
    #     company_dt= test['company_dt']
    
    print("end construct test data: " + str(len(test_data["patentId"])))

    batches_train = dh.batch_iter(train_data, company_dt, total_number, args.batch_size,args.max_len_neighbors,args.max_len_bert,args.level, type='train', shuffle=False)
    batches_val = dh.batch_iter(val_data, company_dt, total_number, args.batch_size,args.max_len_neighbors,args.max_len_bert,args.level, type='val', shuffle=False)
    batches_test = dh.batch_iter(test_data, company_dt, total_number, args.batch_size,args.max_len_neighbors,args.max_len_bert,args.level, type='test', shuffle=False)
    
    
    # Init DHPT model
    dhpt = DHPT(model_name=args.bert_base.format(args.bert_lang),
                bert_emb_size=args.bert_embedding_dim, 
                word_emb_size=args.wordembedding_dim, 
                lstm_hidden_size=args.lstm_dim,
                layers_num=1, 
                layer_class_nums=total_number,  
                hierarchy_tree=hierarchy_data, 
                com_model_way=args.model_way,
                com_feat_type=args.com_feat_type,
                use_hierarchy=args.use_hierarchy,
                use_vertical=args.use_vertical,
                word2vec_emb_init=embedding_matrix,
                use_company=args.use_company,
                use_history= args.use_his_memory,
                use_memory=args.use_memory, 
                use_time=args.use_time,
                use_cnn = args.use_cnn,
                use_bert = args.use_bert,num_filters= args.num_filters, filter_sizes= args.filter_sizes,
                memory_update_at_start=not args.memory_update_at_end,
                use_lstm_text=args.use_lstm_text,
                message_aggregator_type=args.message_agg,
                n_nodes=sum(total_number), 
                node_emb_type=args.node_embedding_module,
                message_dim=args.message_dim, 
                message_function=args.message_function, 
                memory_dim=args.memory_dim,
                memory_updater_type=args.memory_updater,
                raw_node_features=raw_node_features,
                max_len_neighbors=args.max_len_neighbors,
                labels_num=total_number[args.level - 1],
                level = args.level,
                weight = args.weight,
                dropout=args.dropout_rate,
                device=device ).to(device)
    
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(dhpt.parameters(), lr=args.learning_rate)

    # Generate batches
    keys = ['group']
    
    epochs = args.epochs
    for epoch in range(epochs):
        # if epoch > 0:
        #     break
        logger.info("Start epoch:{} at {} ".format(epoch, datetime.datetime.now()))
        results = {k: {} for k in keys}
        # Reinitialize memory of the model at the start of each epoch
        if args.use_memory and args.use_company:
           dhpt.histories.memory.__init_memory__()
        dhpt.train()
        # Training loop. For each batsch...
        total_sample_numbers, total_loss = 0, 0

        batch_range = tqdm(enumerate(batches_train))
        for idx, batch_train in batch_range:
            # if idx > 2:
            #     break
            x, truths = batch_train
            gpu_word_text, texts, labels = [], [], []
            com_num,com_not_initial,com_label_feats,com_word_text_feats, com_bert_text_feats, \
            com_bert_text_masks, com_bert_text_token_type_ids,com_word_text_len,com_bert_text_len,com_label_len,\
            com_publication_date,com_ts, bert_text_input,bert_text_mask,bert_text_token_type_ids,word_inputs,publicationDate,ts = x

            bert_text_input = bert_text_input.to(device, dtype=torch.long)
            bert_text_mask = bert_text_mask.to(device, dtype=torch.long)
            bert_text_token_type_ids = bert_text_token_type_ids.to(device, dtype=torch.long)
            for elemt in word_inputs:
                gpu_word_text.append(elemt.to(device))
            gpu_word_text = pad_sequence(gpu_word_text, batch_first=True, padding_value=0).to(device)
            
            total_sample_numbers += len(bert_text_input)
            
            local_logits = dhpt(bert_text_input, bert_text_mask, bert_text_token_type_ids,gpu_word_text
                               ,publicationDate,ts,
                                {'text': com_word_text_feats,'com_date':com_publication_date,'ts':torch.Tensor(com_ts).to(device,dtype=torch.float), 'label': com_label_feats},
                                {'text': com_word_text_len,'com_num':com_num,'label': com_label_len}, com_not_initial)
            
            loss = 0
            truths = [truth.to(device) for truth in truths]
            # print(truths[0].shape)
            for truth, pred in zip(truths, local_logits):
                loss += criterion(pred, truth)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_avg_loss = loss.cpu().data.numpy()
            total_loss += batch_avg_loss
            
            for j, (truth, pred, key) in enumerate(zip(truths, local_logits, keys)):
                scores = get_metric(y_true=truth.detach().cpu(), y_pred=pred.detach().cpu(), idx=j)
                for metric_i, metric_value in scores.items():
                    if metric_i not in results[key]:
                        results[key][metric_i] = 0
                    results[key][metric_i] += metric_value
            
            batch_range.set_description(f"loss: {batch_avg_loss},{total_loss};")

            if args.use_memory and args.use_company:
                dhpt.histories.memory.detach_memory()
                if args.use_his_memory:
                    dhpt.histories.his_memory.detach_his_memory()
                    current_min_date_key = (max(publicationDate))[0:4] # 2010
                    if int(dhpt.histories.his_memory.max_date_key) < int(current_min_date_key):
                        key = dhpt.histories.his_memory.max_date_key - 2010
                        dhpt.histories.his_memory.set_his_memory(key, dhpt.histories.memory.memory.data)
                        history_mem = dhpt.histories.his_memory.his_memory.clone()
                        np.save(cur_path + f"/IPC/{args.US}/{model_type}/"+str(dhpt.histories.his_memory.max_date_key), history_mem.cpu())
                        dhpt.histories.his_memory.max_date_key = int(current_min_date_key)

        
        for k, k_items in results.items():
            for met_key, met_value in k_items.items():
                results[k][met_key] /= total_sample_numbers
        
        epoch_time = time.time() - start_epoch
        logger.info('epoch: {} took {:.2f}s'.format(epoch, epoch_time))

        save_results(cur_path + '/results/'+args.US+'/'+ model_type +'_train.json', results)
        logger.info("Loss in training:{}".format(total_loss / total_sample_numbers))
        logger.info("Metrics in training:{}".format(results))
        print("===================================val===================================")
        
        if args.use_memory and args.use_company:
                train_memory_backup = dhpt.histories.memory.backup_memory()
                dhpt.histories.memory.restore_memory(train_memory_backup)
                if args.use_his_memory:
                   train_his_memory_backup = dhpt.histories.his_memory.backup_his_memory()
                   dhpt.histories.his_memory.restore_his_memory(train_his_memory_backup)
        # Backup memory at the end of training, so later we can restore it and use it for the
        # validation on unseen nodes
        val_scores, valloss = evaluation(dhpt, batches_val, logger, device)
        
        if args.use_memory and args.use_company:
                val_memory_backup = dhpt.histories.memory.backup_memory()
                dhpt.histories.memory.restore_memory(val_memory_backup)
                if args.use_his_memory:
                   val_his_memory_backup = dhpt.histories.his_memory.backup_his_memory()
                   dhpt.histories.his_memory.restore_his_memory(val_his_memory_backup)

        save_results(cur_path + '/results/'+args.US+'/'+model_type+'validate.json', val_scores)
        validate_precision_list = []
        for level_key, level_value in val_scores.items():
            for metric_key, metric_value in level_value.items():
                if metric_key.startswith("precision_"):
                    validate_precision_list.append(metric_value)
        validate_precision = np.mean(validate_precision_list)
        ifstop, ifimprove = early_stopper.early_stop_check(validate_precision)
        if ifstop:
            logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
            logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
            best_model_path = get_checkpoint_path(early_stopper.best_epoch)
            dhpt.load_state_dict(torch.load(best_model_path))
            logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
            print("===================================test===================================")
            test_scores, testloss = evaluation(dhpt, batches_test, logger, device, string='test')
            save_results(cur_path + '/results/'+args.US+'/'+model_type+'test.json', test_scores)
            total_epoch_time = time.time() - start_epoch
            logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
            exit(0)
        if ifimprove:
            logger.info("save models at epoch :{}".format(epoch))
            torch.save(
                dhpt.state_dict(),
                get_checkpoint_path(early_stopper.best_epoch))    
        print("===================================test===================================")
        test_scores, testloss = evaluation(dhpt, batches_test, logger, device, string='test')
        save_results(cur_path + '/results/'+args.US+'/'+model_type+'test.json', test_scores)
    total_epoch_time = time.time() - start_epoch
    logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
