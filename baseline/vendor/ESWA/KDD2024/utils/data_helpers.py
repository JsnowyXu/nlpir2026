# -*- coding:utf-8 -*-
import os
import time
import heapq

import dgl
import logging
import json

import jieba
import numpy as np
from collections import OrderedDict, Counter

import pandas as pd
from texttable import Texttable
from gensim.models import word2vec, KeyedVectors
import torch
from torch.utils.data import Dataset, DataLoader,SequentialSampler
from tqdm import tqdm
import transformers
from transformers import BertTokenizer

def _option(pattern):
    """
    Get the option according to the pattern.
    pattern 0: Choose training or restore.
    pattern 1: Choose best or latest checkpoint.

    Args:
        pattern: 0 for training step. 1 for testing step.
    Returns:
        The OPTION.
    """
    if pattern == 0:
        OPTION = input("[Input] Train or Restore? (T/R): ")
        while not (OPTION.upper() in ['T', 'R']):
            OPTION = input("[Warning] The format of your input is illegal, please re-input: ")
    if pattern == 1:
        OPTION = input("Load Best or Latest Model? (B/L): ")
        while not (OPTION.isalpha() and OPTION.upper() in ['B', 'L']):
            OPTION = input("[Warning] The format of your input is illegal, please re-input: ")
    return OPTION.upper()


def logger_fn(name, input_file, level=logging.INFO):
    """
    The Logger.

    Args:
        name: The name of the logger.
        input_file: The logger file path.
        level: The logger level.
    Returns:
        The logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    log_dir = os.path.dirname(input_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # File Handler
    fh = logging.FileHandler(input_file, mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # stream Handler
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    sh.setLevel(logging.INFO)
    logger.addHandler(sh)
    return logger


def tab_printer(args, logger):
    """
    Function to print the logs in a nice tabular format.

    Args:
        args: Parameters used for the model.
        logger: The logger.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    t.add_rows([["Parameter", "Value"]])
    logger.info('\n' + t.draw())


def get_out_dir(option, logger):
    """
    Get the out dir for saving model checkpoints.

    Args:
        option: Train or Restore.
        logger: The logger.
    Returns:
        The output dir for model checkpoints.
    """
    if option == 'T':
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        logger.info("Writing to {0}\n".format(out_dir))
    if option == 'R':
        MODEL = input("[Input] Please input the checkpoints model you want to restore, "
                      "it should be like (1490175368): ")  # The model you want to restore

        while not (MODEL.isdigit() and len(MODEL) == 10):
            MODEL = input("[Warning] The format of your input is illegal, please re-input: ")
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", MODEL))
        logger.info("Writing to {0}\n".format(out_dir))
    return out_dir


def get_model_name():
    """
    Get the model name used for test.

    Returns:
        The model name.
    """
    MODEL = input("[Input] Please input the model file you want to test, it should be like (1490175368): ")

    while not (MODEL.isdigit() and len(MODEL) == 10):
        MODEL = input("[Warning] The format of your input is illegal, "
                      "it should be like (1490175368), please re-input: ")
    return MODEL


def create_prediction_file(output_file, data_id, true_labels, predict_labels, predict_scores):
    """
    Create the prediction file.

    Args:
        output_file: The all classes predicted results provided by network.
        data_id: The data record id info provided by dict <Data>.
        true_labels: The all true labels.
        predict_labels: The all predict labels by threshold.
        predict_scores: The all predict scores by threshold.
    Raises:
        IOError: If the prediction file is not a .json file.
    """
    if not output_file.endswith('.json'):
        raise IOError("[Error] The prediction file is not a json file."
                      "Please make sure the prediction data is a json file.")
    with open(output_file, 'w') as fout:
        data_size = len(predict_labels)
        for i in range(data_size):
            data_record = OrderedDict([
                ('id', data_id[i]),
                ('labels', [int(i) for i in true_labels[i]]),
                ('predict_labels', [int(i) for i in predict_labels[i]]),
                ('predict_scores', [round(i, 4) for i in predict_scores[i]])
            ])
            fout.write(json.dumps(data_record, ensure_ascii=False) + '\n')


def get_onehot_label_threshold(scores, threshold=0.5):
    """
    Get the predicted one-hot labels based on the threshold.
    If there is no predict score greater than threshold, then choose the label which has the max predict score.

    Args:
        scores: The all classes predicted scores provided by network.
        threshold: The threshold (default: 0.5).
    Returns:
        predicted_onehot_labels: The predicted labels (one-hot).
    """
    predicted_onehot_labels = []
    scores = np.ndarray.tolist(scores)
    for score in scores:
        count = 0
        onehot_labels_list = [0] * len(score)
        for index, predict_score in enumerate(score):
            if predict_score >= threshold:
                onehot_labels_list[index] = 1
                count += 1
        if count == 0:
            max_score_index = score.index(max(score))
            onehot_labels_list[max_score_index] = 1
        predicted_onehot_labels.append(onehot_labels_list)
    return predicted_onehot_labels


def get_onehot_label_topk(scores, top_num=1):
    """
    Get the predicted one-hot labels based on the topK.

    Args:
        scores: The all classes predicted scores provided by network.
        top_num: The max topK number (default: 5).
    Returns:
        predicted_onehot_labels: The predicted labels (one-hot).
    """
    predicted_onehot_labels = []
    scores = np.ndarray.tolist(scores)
    for score in scores:
        onehot_labels_list = [0] * len(score)
        max_num_index_list = list(map(score.index, heapq.nlargest(top_num, score)))
        for i in max_num_index_list:
            onehot_labels_list[i] = 1
        predicted_onehot_labels.append(onehot_labels_list)
    return predicted_onehot_labels


def get_label_threshold(scores, threshold=0.5):
    """
    Get the predicted labels based on the threshold.
    If there is no predict score greater than threshold, then choose the label which has the max predict score.
    Note: Only Used in `test_model.py`

    Args:
        scores: The all classes predicted scores provided by network.
        threshold: The threshold (default: 0.5).
    Returns:
        predicted_labels: The predicted labels.
        predicted_scores: The predicted scores.
    """
    predicted_labels = []
    predicted_scores = []
    scores = np.ndarray.tolist(scores)
    for score in scores:
        count = 0
        index_list = []
        score_list = []
        for index, predict_score in enumerate(score):
            if predict_score >= threshold:
                index_list.append(index)
                score_list.append(predict_score)
                count += 1
        if count == 0:
            index_list.append(score.index(max(score)))
            score_list.append(max(score))
        predicted_labels.append(index_list)
        predicted_scores.append(score_list)
    return predicted_labels, predicted_scores


def get_label_topk(scores, top_num=1):
    """
    Get the predicted labels based on the topK.
    Note: Only Used in `test_model.py`

    Args:
        scores: The all classes predicted scores provided by network.
        top_num: The max topK number (default: 5).
    Returns:
        The predicted labels.
    """
    predicted_labels = []
    predicted_scores = []
    scores = np.ndarray.tolist(scores)
    for score in scores:
        score_list = []
        index_list = np.argsort(score)[-top_num:]
        index_list = index_list[::-1]
        for index in index_list:
            score_list.append(score[index])
        predicted_labels.append(np.ndarray.tolist(index_list))
        predicted_scores.append(score_list)
    return predicted_labels, predicted_scores


def create_metadata_file(word2vec_file, output_file):
    """
    Create the metadata file based on the corpus file (Used for the Embedding Visualization later).

    Args:
        word2vec_file: The word2vec file.
        output_file: The metadata file path.
    Raises:
        IOError: If word2vec model file doesn't exist.
    """
    if not os.path.isfile(word2vec_file):
        raise IOError("[Error] The word2vec file doesn't exist.")

    wv = KeyedVectors.load(word2vec_file, mmap='r')
    word2idx = dict([(k, v.index) for k, v in wv.vocab.items()])
    word2idx_sorted = [(k, word2idx[k]) for k in sorted(word2idx, key=word2idx.get, reverse=False)]

    with open(output_file, 'w+') as fout:
        for word in word2idx_sorted:
            if word[0] is None:
                print("[Warning] Empty Line, should replaced by any thing else, or will cause a bug of tensorboard")
                fout.write('<Empty Line>' + '\n')
            else:
                fout.write(word[0] + '\n')


def load_word2vec_matrix(word2vec_file):
    """
    Get the word2idx dict and embedding matrix.

    Args:
        word2vec_file: The word2vec file.
    Returns:
        word2idx: The word2idx dict.
        embedding_matrix: The word2vec model matrix.
    Raises:
        IOError: If word2vec model file doesn't exist.
    """
    if not os.path.isfile(word2vec_file):
        raise IOError("[Error] The word2vec file doesn't exist. ")

    model = word2vec.Word2Vec.load(word2vec_file, mmap='r')
    print(model)
    word2idx = OrderedDict({"PAD_": 0, "_UNK": 1})
    embedding_size = model.wv.vector_size
    count = 2
    for k, v in model.wv.vocab.items():
        word2idx[k] = count
        # print(k, count)
        count += 1
    vocab_size = len(word2idx)
    print(len(model.wv.vocab.keys()), vocab_size)
    embedding_matrix = torch.zeros([vocab_size, embedding_size])
    for key, value in word2idx.items():
        if key == 'PAD_':
            continue
        elif key == "_UNK":
            embedding_matrix[value] = torch.randn(embedding_size)
        else:
            embedding_matrix[value] = torch.tensor(model.wv[key])
    return word2idx, embedding_matrix


def load_data_and_labels(args, input_file, word2idx: dict, level=3, company_maps=None, company_ids=0,
                         patent_maps=None, patent_ids=1, company_dts=None, type='train'):
    """
    Load research data from files, padding sentences and generate one-hot labels.

    Args:
        args: The arguments.
        input_file: Datasets for train/valid/test.
        word2idx: The word2idx dict.
        level: The specific hierarchy.
    Returns:
        The dictionary of <Data> (includes the word2vec content index, bert token index and mask, company patent, patent records )
    Raises:
        IOError: If word2vec model file doesn't exist
    """

    def _token_to_index(x: list):
        result = []
        for item in x:
            if item not in word2idx.keys():
                result.append(word2idx['_UNK'])
                # result.append(word2idx['<UNK>'])
            else:
                word_idx = word2idx[item]
                result.append(word_idx)
        return torch.tensor(result, dtype=torch.long)

    Data = dict()
    tokenizer = BertTokenizer.from_pretrained(args.bert_base.format(args.bert_lang))

    company_id = company_ids
    company_map = company_maps if company_maps else dict()
    patent_id = patent_ids
    patent_map = patent_maps if patent_maps else dict()
    company_dt = company_dts if company_dts else {'patents': {}, 'company': {}}

    dataset = pd.read_csv(input_file)
    dataset.sort_values(by='publicationDate', inplace=True)
   
    Data['bert_content_index'] = []
    Data['bert_mask'] = []
    Data['bert_token_type_ids'] = []
    Data['word_content_index'] = []
    Data['ipc_codes'] = []
    Data['company'] = []
    Data['patentId'] = []
    Data['publicationDate'] = []
    Data['ts'] = []

    multi_com = 0
    multiclass = 0
    for id, patent in tqdm(dataset.iterrows()):
        if args.US == 'US':
            content = patent['summary'].lower().split()[:args.max_len_bert]# the content input to bert
            word_content = patent['summary'].split()[:args.max_len_word]# the content input to word2vec
        else:
            content =  str(patent['summary'])[:args.max_len_bert]
            word_content = (" ".join(jieba.cut(patent['summary'], cut_all=False))).split()[:args.max_len_word]
        # bert tokenizer
        inputs = tokenizer.encode_plus(
            content,
            add_special_tokens=True,
            max_length=args.max_len_bert,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]
        bert_content_id = torch.tensor(ids, dtype=torch.long)
        mask = torch.tensor(mask, dtype=torch.long)
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
        Data['bert_content_index'].append(bert_content_id)
        Data['bert_mask'].append(mask)
        Data['bert_token_type_ids'].append(token_type_ids)
        word_content_id = _token_to_index(word_content)
        Data['word_content_index'].append(word_content_id)
        
        # IPC code list for current patent
        if level == 3:
            ipc_codes = list(map(int, patent['codeIIIId'].split(";")))
        elif level == 2:
            ipc_codes = list(map(int, patent['codeIIId'].split(";")))
        elif level == 1:
            ipc_codes = list(map(int, patent['codeIId'].split(";")))
        if len(ipc_codes) > 1:
            multiclass = multiclass + 1 
        Data['ipc_codes'].append(ipc_codes)
        
        # calculate total patent number and generate patent dict and patent list in Data
        if patent['patentId'] not in patent_map:
            patent_map[patent['patentId']] = patent_id
            patent_id += 1
        Data['patentId'].append(patent_map[patent['patentId']])
        Data['publicationDate'].append(patent['publicationDate'])
        Data['ts'].append(patent['ts'])
        # generate company related text and label list
        patent_com = []

        if len(patent['orgID'].split(";")) > 1:
                multi_com +=1
        for com in patent['orgID'].split(";"):
            if com not in company_map:
                company_map[com] = company_id
                company_id += 1
            patent_com.append(company_map[com])
            if company_map[com] not in company_dt['company']:
                company_dt['company'][company_map[com]] = {"pid": [],'publicationDate':[],'ts':[],"word_text":[], "bert_text": [], 'label': [],
                                                                    'word_text_len':[],'bert_text_len': [],'bert_mask':[], 'bert_token_type_ids':[],
                                                                    'label_len': []}
                
            # patent number of a company of the current patent before the application date of current patent. 截止当前专利申请日期之前，该专利申请人已经发表的专利数量
            if  patent_map[patent['patentId']] not in company_dt['patents']:
                company_dt['patents'][patent_map[patent['patentId']]] = { }
            company_dt['patents'][patent_map[patent['patentId']]][company_map[com]] = len(company_dt['company'][company_map[com]]["pid"])
            company_dt['company'][company_map[com]]['pid'].append(patent_map[patent['patentId']])
            company_dt['company'][company_map[com]]['publicationDate'].append(patent['publicationDate'])
            company_dt['company'][company_map[com]]['ts'].append(torch.tensor(patent['ts']))
            # bert text/mask/token_type_ids and text length
            company_dt['company'][company_map[com]]['bert_text'].append(bert_content_id)
            company_dt['company'][company_map[com]]['bert_text_len'].append(len(bert_content_id))
            company_dt['company'][company_map[com]]['bert_mask'].append(mask)
            company_dt['company'][company_map[com]]['bert_token_type_ids'].append(token_type_ids)
            # word2vec text and text length
            company_dt['company'][company_map[com]]['word_text'].append(word_content_id)
            company_dt['company'][company_map[com]]['word_text_len'].append(len(word_content_id))
            # label list and label length
            company_dt['company'][company_map[com]]['label'].append(torch.tensor(ipc_codes, dtype=torch.long))
            company_dt['company'][company_map[com]]['label_len'].append(len(ipc_codes))
        Data['company'].append(patent_com)
    print(f'total multi-com:{multi_com}')
    return Data, company_map, company_id, patent_map, patent_id, company_dt

def align_com_feature(bert_com_feats):
    text_feats = []
    for item in bert_com_feats:
        for items in item:
            text_feats.append(items)
    bert_com_feats = text_feats
    return bert_com_feats

def collate_fn(data):
    """
    Args:
        data (List[Tuple]): len(data) = batch_size
            list[0] (int):historical patent count of a com
            list[1] (tuple): company_feat
            list[2~7] :patent feat
            list[8]:ipc codes
    Returns:
        inputs (Tensor): shape (batch_size, word_lens)
        truth (Tensor): shape (batch_size, label_number)
    """
    input, truth = [], []
    # data:company list, company feature list, 'bert_content_index','bert_mask','bert_token_type_ids','word_content_index', 'ipc_codes'
    for i, items in enumerate(zip(*data)):
        if i > 8:
            items = torch.cat(items, dim=0) # truth label feature: k-hot embedding
            truth.append(items)
        elif i == 2:
        # 有时间需要进一步优化下代码
        # company_feat.append((label_feats,word_text_feats, bert_text_feats, bert_text_masks, bert_text_token_type_ids,word_text_len,bert_text_len,label_len,publication_date,ts, not_initial))
            label_feats,word_text_feats, bert_text_feats, \
                bert_text_masks, bert_text_token_type_ids,word_text_len,bert_text_len,label_len,publication_date,ts=[],[],[],[],[],[],[],[],[],[]   
            # label_feats_list,word_text_feats_list, bert_text_feats_list, \
                # bert_text_masks_list, bert_text_token_type_ids_list,word_text_len_list,bert_text_len_list,label_len_list,publication_date_list,ts_list, not_initial_list=[],[],[],[],[],[],[],[],[],[],[],[],[],[]
            for i in range(len(items)):
               for j in range(len(items[i])):
                    for k in range(len(items[i][j])):
                        values = items[i][j][k]
                        if k == 0:
                            label_feats.append(values)
                        elif k==1:
                            word_text_feats.append(values)
                        elif k==2:
                            bert_text_feats.append(values)
                        elif k==3:
                            bert_text_masks.append(values)
                        elif k==4:
                            bert_text_token_type_ids.append(values)
                        elif k==5:
                            word_text_len.append(values)
                        elif k==6:
                            bert_text_len.append(values)
                        elif k==7:
                            label_len.append(values)
                        elif k==8:
                            publication_date.append(values)
                        elif k==9:
                            ts.append(values)
                        # elif k==10:
                        #     not_initial.append(values)
            input.append(align_com_feature(label_feats))
            input.append(align_com_feature(word_text_feats))
            input.append(align_com_feature(bert_text_feats))      
            input.append(align_com_feature(bert_text_masks))
            input.append(align_com_feature(bert_text_token_type_ids))
            input.append(align_com_feature(word_text_len))
            input.append(align_com_feature(bert_text_len))
            input.append(align_com_feature(label_len))
            input.append(align_com_feature(publication_date))
            input.append(align_com_feature(ts))
            # input.append(torch.cat(not_initial))
        elif i > 2 and i < 6: # bert features
            items = torch.cat(items, dim=0)
            input.append(items)
        elif i == 1:
            init = [sum(item) for item in items]
            flat_list = [item for sublist in init for item in sublist]  
            input.append(np.array(flat_list))
        else:
            input.append(items) # patent features
    return input, truth


class My_dataset(Dataset):
    def __init__(self, data, company_dts, total_num,max_len_neighbors,max_len_bert,level, type='train'):
        super().__init__()
        self.data = data
        self.level = level
        self.keys = ['bert_content_index','bert_mask','bert_token_type_ids','word_content_index','publicationDate','ts', 'ipc_codes']
        self.total_num = total_num
        self.length = len(self.data[self.keys[0]])
        self.companies = company_dts  # "patents":(idx, text, ipc_codes); "company":[id1, id2, ...]
        self.type = type
        self.max_len_neighbors = max_len_neighbors
        self.max_len_bert = max_len_bert
    def _create_onehot_labels(self, labels_index, num_labels):
        label = torch.zeros([num_labels])
        # print(labels_index)
        if isinstance(labels_index, torch.Tensor):
            label[labels_index] = 1
        else:
            index_lb = torch.tensor(labels_index, dtype=torch.long)
            label[index_lb] = 1
        return label.unsqueeze(dim=0)

    def __getitem__(self, index):
        item_dt = []
        pid = self.data['patentId'][index]
        company = []
        com_not_initial = []
        company_feat = []
        for com in self.companies['patents'][pid].keys():
            # 当前专利发表时间之前该公司发表专利数量
            index_com_count = self.companies['patents'][pid][com]
            not_initial = torch.ones(1, dtype=torch.long)
            if index_com_count == 0:
                not_initial[0] = 0
                nodes = [0]
                bert_text_nodes = [torch.tensor([0]* self.max_len_bert)]
                bert_masks = [torch.tensor([0] * self.max_len_bert)]
                bert_token_type_ids = [torch.tensor([0] * self.max_len_bert)]
                bert_text_len = [1]
                label_nodes = [torch.tensor([0])]
                label_len = [1]
                word_text_nodes = [torch.tensor([0])]
                word_text_len = [1]
                publication_date = [0]
                ts = [torch.tensor([0])]
            else:
                start = max(index_com_count - self.max_len_neighbors, 0)
                nodes = self.companies['company'][com]['pid'][start:index_com_count]
                word_text_nodes = self.companies['company'][com]['word_text'][start:index_com_count]
                word_text_len = self.companies['company'][com]['word_text_len'][start:index_com_count]

                bert_text_nodes = self.companies['company'][com]['bert_text'][start:index_com_count]
                bert_masks= self.companies['company'][com]['bert_mask'][start:index_com_count]
                bert_token_type_ids =  self.companies['company'][com]['bert_token_type_ids'][start:index_com_count]
                bert_text_len = self.companies['company'][com]['bert_text_len'][start:index_com_count]
               
                label_nodes = self.companies['company'][com]['label'][start:index_com_count]
                label_len = self.companies['company'][com]['label_len'][start:index_com_count]
                
                publication_date = self.companies['company'][com]['publicationDate'][start:index_com_count]
                ts = self.companies['company'][com]['ts'][start:index_com_count]
            word_text_feats = word_text_nodes
            label_feats = label_nodes
            bert_text_feats = bert_text_nodes
            bert_text_masks = bert_masks
            bert_text_token_type_ids = bert_token_type_ids
            company.append(len(nodes)) # actual patents count for current com
            com_not_initial.append(not_initial)
            company_feat.append((label_feats,word_text_feats, bert_text_feats, bert_text_masks, bert_text_token_type_ids,word_text_len,bert_text_len,label_len,publication_date,ts))
        item_dt.append(company)
        item_dt.append(com_not_initial)
        item_dt.append(company_feat)

        for i, key in enumerate(self.keys):
            # self.keys = ['bert_content_index','bert_mask','bert_token_type_ids','word_content_index','publicationDate','ts', 'ipc_codes']
            if i > 5:
                item_dt.append(self._create_onehot_labels(self.data[key][index], self.total_num[self.level - 1]))
            elif i >= 3:
                item_dt.append(self.data[key][index])
            else:
                item_dt.append(self.data[key][index].unsqueeze(dim=0))
        return item_dt

    def __len__(self):
        return self.length

def batch_iter(data: [torch.Tensor], company_dts, num_classes_list, batch_size,max_len_neighbors,max_len_bert,level = 3, type='train', shuffle=False):
    """
    含有 yield 说明不是一个普通函数，是一个 Generator.
    函数效果：对 data，一共分成 num_epochs 个阶段（epoch），在每个 epoch 内，如果 shuffle=True，就将 data 重新洗牌，
    批量生成 (yield) 一批一批的重洗过的 data，每批大小是 batch_size，一共生成 int(len(data)/batch_size)+1 批。

    Args:
        data: zip(data['pad_seqs'], data['section'], data['subsection'], data['ipc_codes'],
               data['subgroup'], data['onehot_labels'], data['labels'])
        batch_size: The size of the data batch.
        num_epochs: The number of epochs.
        shuffle: Shuffle or not (default: True).
    Returns:
        A batch iterator for data set.
    """
    data_set = My_dataset(data, company_dts, num_classes_list,max_len_neighbors,max_len_bert,level, type=type)
    sampler = SequentialSampler(data_set)
    data_loader = DataLoader(data_set, batch_size=batch_size, collate_fn=collate_fn, sampler=sampler)
    return data_loader


def load_adjacent(data):
    dt = np.load(data).astype('float32')
    torch_dt = torch.from_numpy(dt)
    # print(torch_dt)
    return torch_dt


def load_hierarchy_tree(raw_path, data_type, hierarchy_names) -> [torch.Tensor, []]:
    hierarchy_dt = []
    total_number_pre = []
    total_number_aft = [0]
    for hier_name in hierarchy_names:
        hier_path = raw_path.format(data_type, hier_name)
        hier_data = pd.read_csv(hier_path)
        data = hier_data.values
        total_number_pre.append(max(data[:, 0]) + 1)
        total_number_aft.append(max(data[:, 1]) + 1)
        hierarchy_dt.append(torch.tensor(data, dtype=torch.int))
        # print("hierarchy size for layer {}:{}".format(hier_name, hierarchy_dt[-1].shape))
    total_number_aft.append(0)
    total_number_pre.append(632)

    # total_number_pre.append(max(data[:, 1]) + 1)

    total_num = []
    for low_count, high_count in zip(total_number_pre, total_number_aft):
        count = max(low_count, high_count)
        total_num.append(count)

    # print("classification codes in hierarchical tree is:{}".format(total_num))

    hierarchy_matrix = []
    pre_num = total_num[0]
    for aft_id in torch.arange(1, len(total_num)):
        aft_num = total_num[aft_id]
        matrix = torch.zeros(pre_num, aft_num)
        for line in hierarchy_dt[aft_id - 1]:
            matrix[line[0]][line[1]] = 1.0
        hierarchy_matrix.append(matrix)
        pre_num = aft_num
    return hierarchy_matrix, total_num


if __name__ == '__main__':
    print("main")
