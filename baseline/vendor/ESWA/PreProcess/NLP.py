from gensim.models import word2vec, KeyedVectors
import pandas as pd
import jieba
import json
import os
import numpy as np
from collections import OrderedDict
import torch


dirpath = os.path.dirname(os.path.realpath(__file__)) +'/'
print (dirpath)

with open(dirpath + 'config/config.json', 'r') as f:
    data_config = json.load(f)

def train_modelCN():
    def train_savemodel(sentences, model_path, wv_path):
        model = word2vec.Word2Vec(sentences, min_count=5, workers=3,
                                  sg=1,  # 使用 skip-gram算法
                                  hs=1,  # 使用分层softmax
                                  negative=0)  # 不使用负采样和噪音
        model.save(model_path)
        model.wv.save_word2vec_format(wv_path, binary=True)

    raw_datas = pd.read_csv(dirpath + "./data/CN-filter/new-CN/train-val-test/train.csv", encoding="utf-8", low_memory=False)
    raw_datas = raw_datas.dropna()
    raw_sentences = raw_datas.summary.values
    print(raw_sentences)
    sentences = [(" ".join(jieba.cut(s, cut_all=False))).split() for s in raw_sentences]
    print(sentences[:10])
    train_savemodel(sentences,  dirpath + './NLPModel/word2vec-CN-kdd-200K.kv', dirpath +'./NLPModel/word2vec.kv.vectors-CN-kdd-200k.npy')


def train_model_US():
    def train_savemodel(sentences, model_path, wv_path):
        model = word2vec.Word2Vec(sentences, min_count=5, workers=3,
                                  sg=1,  # 使用 skip-gram算法
                                  hs=1,  # 使用分层softmax
                                  negative=0)  # 不使用负采样和噪音
        model.save(model_path)
        model.wv.save_word2vec_format(wv_path, binary=True)

    raw_datas = pd.read_csv(dirpath + "./data/US-filter/new-US/train-val-test/train.csv", encoding="utf-8", low_memory=False)
    raw_datas = raw_datas.dropna()
    raw_sentences = raw_datas.summary.values
    print(raw_sentences)
    sentences = [s.lower().split() for s in raw_sentences]
    train_savemodel(sentences,dirpath+ './NLPModel/word2vec-US-kdd-200K.kv', dirpath+'./NLPModel/word2vec.kv.vectors-US-kdd-200k.npy')

def train_IPC_CN():
    def train_savemodel(sentences, model_path, wv_path):
        model = word2vec.Word2Vec(sentences, min_count=1, workers=3,
                                  sg=1,  # 使用 skip-gram算法
                                  hs=1,  # 使用分层softmax
                                  negative=0)  # 不使用负采样和噪音
        model.save(model_path)
        model.wv.save_word2vec_format(wv_path, binary=True)
    raw_ipc1 = pd.read_csv(dirpath + "./IPC/IPCSection.csv", sep=' ', encoding="utf-8", low_memory=False)
    raw_ipc2 = pd.read_csv(dirpath + "./IPC/IPCClass.csv", sep=' ', encoding="utf-8", low_memory=False)
    raw_ipc3 = pd.read_csv(dirpath + "./IPC/IPCSubClass.csv", sep=' ')

    raw_sentences = list(raw_ipc1.Name.values)
    raw_sentences.extend(list(raw_ipc2.Name.values))
    raw_sentences.extend(list(raw_ipc3.Name.values))
    sentences = [(" ".join(jieba.cut(s, cut_all=False))).split() for s in raw_sentences]

    print(raw_sentences)
    print(sentences[:10])
    train_savemodel(sentences,dirpath+ './NLPModel/word2vec-CN-IPC.kv', dirpath+'./NLPModel/word2vec.kv.vectors-CN-IPC.npy')


def train_IPC_US():
    def train_savemodel(sentences, model_path, wv_path):
        model = word2vec.Word2Vec(sentences, min_count=1, workers=3,
                                  sg=1,  # 使用 skip-gram算法
                                  hs=1,  # 使用分层softmax
                                  negative=0)  # 不使用负采样和噪音
        model.save(model_path)
        model.wv.save_word2vec_format(wv_path, binary=True)
    raw_ipc1 = pd.read_csv(dirpath + "./IPC/US/IPCSection.csv", sep=',', encoding="utf-8", low_memory=False)
    raw_ipc2 = pd.read_csv(dirpath + "./IPC/US/IPCClass.csv", sep=',', encoding="utf-8", low_memory=False)
    raw_ipc3 = pd.read_csv(dirpath + "./IPC/US/IPCSubClass.csv", sep=',')

    raw_sentences = list(raw_ipc1.Name.values)
    raw_sentences.extend(list(raw_ipc2.Name.values))
    raw_sentences.extend(list(raw_ipc3.Name.values))
    sentences = [(" ".join(jieba.cut(s, cut_all=False))).split() for s in raw_sentences]

    print(raw_sentences)
    print(sentences[:10])
    train_savemodel(sentences,dirpath+ './NLPModel/word2vec-US-IPC.kv', dirpath+'./NLPModel/word2vec.kv.vectors-US-IPC.npy')


def generate_raw_node_feature(keys):
    def _token_to_index(words: list):
        result = []
        for item in words:
            if item not in word2idx.keys():
                result.append(embedding_matrix[word2idx['_UNK']])
            else:
                word_idx = embedding_matrix[word2idx[item]]
                result.append(word_idx)
        # test= np.mean(torch.tensor(result, dtype=torch.long), axis = 0)
        sentence_vector = np.mean(np.stack(result, axis=0), axis=0)  
        return torch.tensor(sentence_vector, dtype=torch.float)
    model = word2vec.Word2Vec.load('/home/xuxj/PatentLabelling/patent-classification/PreProcess/NLPModel/word2vec-CN-IPC.kv', mmap='r')
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
    OUT_NODE_FEAT = '/home/xuxj/PatentLabelling/patent-classification/KDD2024/memories/CN/raw_node_features.npy'
    code_path = data_config[keys]['hierarchy_IPC']
    raw_ipc1 = pd.read_csv(dirpath + "./IPC/IPCSection.csv", sep=' ', encoding="utf-8", low_memory=False)
    raw_ipc2 = pd.read_csv(dirpath + "./IPC/IPCClass.csv", sep=' ', encoding="utf-8", low_memory=False)
    raw_ipc3 = pd.read_csv(dirpath + "./IPC/IPCSubClass.csv", sep=' ')

    i = 1
    ipc_matrix1 = []
    ipc_matrix2 = []
    ipc_matrix3 = []

    for path in code_path:
        codes = pd.read_csv(dirpath+ path)
        if i == 1:
            ipc_matrix1 = torch.zeros([len(codes), embedding_size], dtype=torch.float)
            ipc = pd.merge(left=codes, right=raw_ipc1, left_on='codeI', right_on='Class')
        elif i==2:
            ipc_matrix2 = torch.zeros([len(codes), embedding_size], dtype=torch.float)
            ipc = pd.merge(left=codes, right=raw_ipc2, left_on='codeII', right_on='Class')
        elif i==3:
            ipc_matrix3 = torch.zeros([len(codes), embedding_size], dtype=torch.float)
            ipc = pd.merge(left=codes, right=raw_ipc3, left_on='codeIII', right_on='Class')
        # IPC 从高到底
        for id, patent in ipc.iterrows():
            ipctext = (" ".join(jieba.cut(patent['Name'], cut_all=False))).split()
            if i == 1:
                ipc_matrix1[id] = _token_to_index(ipctext)
            elif i == 2:
                ipc_matrix2[id] = _token_to_index(ipctext)
            else:
                ipc_matrix3[id] = _token_to_index(ipctext)   
        i = i + 1
    total_len = len(ipc_matrix1)+len(ipc_matrix2)+len(ipc_matrix3)
    rand_feat = np.zeros((10, total_len, 100))
    for j in range(10): 
        rand_feat[j,:len(ipc_matrix1),:] = ipc_matrix1
        rand_feat[j,len(ipc_matrix1):len(ipc_matrix1)+len(ipc_matrix2),:] = ipc_matrix2
        rand_feat[j,len(ipc_matrix1)+len(ipc_matrix2):total_len,:] = ipc_matrix3

    # history_memory['2010'] = OUT_NODE_FEAT 
    np.save(OUT_NODE_FEAT, rand_feat)
    # with open(OUT_MEMORY_PATH, 'w') as file_obj:
    #     json.dump(history_memory, file_obj)

def generate_raw_node_US_feature(keys):
    def _token_to_index(words: list):
        result = []
        for item in words:
            if item not in word2idx.keys():
                result.append(embedding_matrix[word2idx['_UNK']])
            else:
                word_idx = embedding_matrix[word2idx[item]]
                result.append(word_idx)
        # test= np.mean(torch.tensor(result, dtype=torch.long), axis = 0)
        sentence_vector = np.mean(np.stack(result, axis=0), axis=0)  
        return torch.tensor(sentence_vector, dtype=torch.float)
    model = word2vec.Word2Vec.load('/home/xuxj/PatentLabelling/patent-classification/PreProcess/NLPModel/word2vec-US-IPC.kv', mmap='r')
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
    OUT_NODE_FEAT = '/home/xuxj/PatentLabelling/patent-classification/KDD2024/memories/US/raw_node_features.npy'
    code_path = data_config[keys]['hierarchy_IPC']
    raw_ipc1 = pd.read_csv(dirpath + "./IPC/US/IPCSection.csv", sep=',', encoding="utf-8", low_memory=False)
    raw_ipc2 = pd.read_csv(dirpath + "./IPC/US/IPCClass.csv", sep=',', encoding="utf-8", low_memory=False)
    raw_ipc3 = pd.read_csv(dirpath + "./IPC/US/IPCSubClass.csv", sep=',')

    i = 1
    ipc_matrix1 = []
    ipc_matrix2 = []
    ipc_matrix3 = []

    for path in code_path:
        codes = pd.read_csv(dirpath+ path)
        if i == 1:
            ipc_matrix1 = torch.zeros([len(codes), embedding_size], dtype=torch.float)
            ipc = pd.merge(left=codes, right=raw_ipc1, left_on='codeI', right_on='Class')
        elif i==2:
            ipc_matrix2 = torch.zeros([len(codes), embedding_size], dtype=torch.float)
            ipc = pd.merge(left=codes, right=raw_ipc2, left_on='codeII', right_on='Class')
        elif i==3:
            ipc_matrix3 = torch.zeros([len(codes), embedding_size], dtype=torch.float)
            ipc = pd.merge(left=codes, right=raw_ipc3, left_on='codeIII', right_on='Class')
        # IPC 从高到底
        for id, patent in ipc.iterrows():
            ipctext = (" ".join(jieba.cut(patent['Name'], cut_all=False))).split()
            if i == 1:
                ipc_matrix1[id] = _token_to_index(ipctext)
            elif i == 2:
                ipc_matrix2[id] = _token_to_index(ipctext)
            else:
                ipc_matrix3[id] = _token_to_index(ipctext)   
        i = i + 1
    total_len = len(ipc_matrix1)+len(ipc_matrix2)+len(ipc_matrix3)
    rand_feat = np.zeros((10, total_len, 100))
    for j in range(10): 
        rand_feat[j,:len(ipc_matrix1),:] = ipc_matrix1
        rand_feat[j,len(ipc_matrix1):len(ipc_matrix1)+len(ipc_matrix2),:] = ipc_matrix2
        rand_feat[j,len(ipc_matrix1)+len(ipc_matrix2):total_len,:] = ipc_matrix3

    # history_memory['2010'] = OUT_NODE_FEAT 
    np.save(OUT_NODE_FEAT, rand_feat)
    # with open(OUT_MEMORY_PATH, 'w') as file_obj:
    #     json.dump(history_memory, file_obj)

def generate_raw_node_IN_feature(keys):
    def _token_to_index(words: list):
        result = []
        for item in words:
            if item not in word2idx.keys():
                result.append(embedding_matrix[word2idx['_UNK']])
            else:
                word_idx = embedding_matrix[word2idx[item]]
                result.append(word_idx)
        # test= np.mean(torch.tensor(result, dtype=torch.long), axis = 0)
        sentence_vector = np.mean(np.stack(result, axis=0), axis=0)  
        return torch.tensor(sentence_vector, dtype=torch.float)
    model = word2vec.Word2Vec.load('/home/xuxj/PatentLabelling/patent-classification/PreProcess/NLPModel/word2vec-US-IPC.kv', mmap='r')
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
    OUT_NODE_FEAT = '/home/xuxj/PatentLabelling/patent-classification/KDD2024/memories/IN/raw_node_features.npy'
    code_path = data_config[keys]['hierarchy_IPC']
    raw_ipc1 = pd.read_csv(dirpath + "./IPC/US/IPCSection.csv", sep=',', encoding="utf-8", low_memory=False)
    raw_ipc2 = pd.read_csv(dirpath + "./IPC/US/IPCClass.csv", sep=',', encoding="utf-8", low_memory=False)
    raw_ipc3 = pd.read_csv(dirpath + "./IPC/US/IPCSubClass.csv", sep=',')

    i = 1
    ipc_matrix1 = []
    ipc_matrix2 = []
    ipc_matrix3 = []

    for path in code_path:
        codes = pd.read_csv(dirpath+ path)
        if i == 1:
            ipc_matrix1 = torch.zeros([len(codes), embedding_size], dtype=torch.float)
            ipc = pd.merge(left=codes, right=raw_ipc1, left_on='codeI', right_on='Class')
        elif i==2:
            ipc_matrix2 = torch.zeros([len(codes), embedding_size], dtype=torch.float)
            ipc = pd.merge(left=codes, right=raw_ipc2, left_on='codeII', right_on='Class')
        elif i==3:
            ipc_matrix3 = torch.zeros([len(codes), embedding_size], dtype=torch.float)
            ipc = pd.merge(left=codes, right=raw_ipc3, left_on='codeIII', right_on='Class')
        # IPC 从高到底
        for id, patent in ipc.iterrows():
            ipctext = (" ".join(jieba.cut(patent['Name'], cut_all=False))).split()
            if i == 1:
                ipc_matrix1[id] = _token_to_index(ipctext)
            elif i == 2:
                ipc_matrix2[id] = _token_to_index(ipctext)
            else:
                ipc_matrix3[id] = _token_to_index(ipctext)   
        i = i + 1
    total_len = len(ipc_matrix1)+len(ipc_matrix2)+len(ipc_matrix3)
    rand_feat = np.zeros((7, total_len, 100))
    for j in range(7): 
        rand_feat[j,:len(ipc_matrix1),:] = ipc_matrix1
        rand_feat[j,len(ipc_matrix1):len(ipc_matrix1)+len(ipc_matrix2),:] = ipc_matrix2
        rand_feat[j,len(ipc_matrix1)+len(ipc_matrix2):total_len,:] = ipc_matrix3

    # history_memory['2010'] = OUT_NODE_FEAT 
    np.save(OUT_NODE_FEAT, rand_feat)
    # with open(OUT_MEMORY_PATH, 'w') as file_obj:
    #     json.dump(history_memory, file_obj)

category_mapping = {  
    'A': '人类生活需要',  
    'B': '作业、运输',  
    'C': '化学、冶金',  
    'D': '纺织、造纸',  
    'E': '固定建筑物',  
    'F': '机械工程、照明、加热、武器、爆破',  
    'G': '物理',
    'H': '电学'  
}  
def process_output(value):  
    # 分割字符串为数组  
    parts = value.split(',')  
    # 处理数组中的每个元素，并替换第一个字母对应的分类  
    processed_parts = [category_mapping.get(part[0].upper(), part) for part in parts]  
    # 使用逗号合并处理后的数组  
    return ','.join(processed_parts)  

def ExtractChatCorpus():
    dataset = pd.read_csv('/home/xuxj/PatentLabelling/patent-classification/PreProcess/data/New-Data/cn-code-all.csv')
    df = dataset.head(10000) 
    df['instruction'] = '作为一个专利管理员，请为输入的内容在后面的分类列表中选择一个或多个适合的分类[人类生活需要，作业、运输，化学、冶金，纺织、造纸，固定建筑物，机械工程、照明、加热、武器、爆破，物理，电学]'
    df_renamed = df.rename(columns={'summary': 'input', 'classCode': 'output'}) 
    df_renamed['output'] = df_renamed['output'].apply(process_output)  
    selected_columns = df_renamed[['instruction','input', 'output']].to_dict('records')  

    # 将选定的列导出为JSON格式  
    # json_data = selected_columns.to_json(orient='records')   
  
    # 使用json.dumps序列化数据，确保中文不被转义  
    json_data = json.dumps(selected_columns, ensure_ascii=False, indent=4)  # indent用于格式化输出，使其更易读  
    
    # 将JSON数据保存到本地文件，指定编码为'utf-8'    
    with open('patent.json', 'w', encoding='utf-8') as f:  
        f.write(json_data)

if __name__ == '__main__':
    key ='new-IN'
    # train_modelCN()
    # train_IPC_CN()
    train_IPC_US()
    # train_model_US()
    # generate_raw_node_US_feature(key)
    generate_raw_node_IN_feature(key)

    # ExtractChatCorpus()
