import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from collections import defaultdict
from modules.memory import Memory
from modules.history_memory import HistoryMemory

from modules.message_aggregator import get_message_aggregator
from modules.message_function import get_message_function
from modules.memory_updater import get_memory_updater
from modules.embedding_module import get_embedding_module
from models.time_encoding import TimeEncode, Time2Vec, PeriodicTimeEncoder
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

# from models.weighted_GCN import weighted_GCN
# 定义LSTM模型  
class LabelEmbeddingLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):  
        super(LabelEmbeddingLSTM, self).__init__()  
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)  
        
    def forward(self, x):  
        # LSTM需要(batch, seq_len, input_size)的输入  
        # 初始化隐藏状态和细胞状态  
        h0 = torch.zeros(1, batch_size, hidden_dim).to(x.device)  # (num_layers * num_directions, batch, hidden_size)  
        c0 = torch.zeros(1, batch_size, hidden_dim).to(x.device)  # (num_layers * num_directions, batch, hidden_size)  
        
        # LSTM的前向传播，得到输出和新的隐藏状态  
        out, _ = self.lstm(x, (h0, c0))  
        
        # 使用最后一个时间步的隐藏状态作为输出  
        # 注意：这里我们取最后一个时间步的输出，其形状为(batch, hidden_dim)  
        last_output = out[:, -1, :]  
        
        # 如果需要，可以将这个形状(batch, hidden_dim)的tensor转换为(batch, 1, hidden_dim)以匹配原始嵌入的维度  
        # last_output = last_output.unsqueeze(1)  
        
        return last_output  

class MLLinear(nn.Module):
    """

    """

    def __init__(self, linear_size, output_size):
        super(MLLinear, self).__init__()
        self.linear = nn.ModuleList(nn.Linear(in_s, out_s)
                                    for in_s, out_s in zip(linear_size[:-1], linear_size[1:]))
        for linear in self.linear:
            nn.init.xavier_uniform_(linear.weight)
        self.output = nn.Linear(linear_size[-1], output_size)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, inputs):
        linear_out = inputs
        for linear in self.linear:
            linear_out = F.relu(linear(linear_out))
        return torch.squeeze(self.output(linear_out), -1)


class HistorySequence(nn.Module):
    def __init__(self, 
                 layer_class_nums, 
                 bert_emb_size,
                 lstm_embedding_label, 
                 lstm_hidden_size, 
                 use_lstm_text,
                 com_feat_type,
                 com_embedding_text, 
                 com_model_way,  
                 n_nodes, 
                 message_dim,
                 memory_dim, 
                 node_emb_type,
                 message_function, 
                 message_aggregator_type,
                 memory_updater_type, 
                 memory_update_at_start,
                 use_memory,
                 use_time,
                 use_history,
                 hierarchy_tree,
                 max_len_neighbors,
                 raw_node_features,
                 time_dim,
                 output_dim, 
                 level,
                 dropout,
                 device
                 ):
        super(HistorySequence, self).__init__()
        self.raw_node_features = torch.from_numpy(raw_node_features.astype(np.float32)).to(device) # 10*L*100
        self.level = level
        # lstm
        self.use_lstm_text = use_lstm_text
        self.lstm_embedding_label = lstm_embedding_label
        self.lstm_hidden_size = lstm_hidden_size
        self.com_feat_type = com_feat_type
        if self.use_lstm_text:
            self.com_input_dim = lstm_hidden_size * 2 # lstm or word2vect dim
        else:
            self.com_input_dim = com_embedding_text
        self.com_model_way = com_model_way

        # his momery
        self.use_history = use_history
        self.memory_dimension = memory_dim
        self.n_nodes = n_nodes
        self.nodes_feat_dim = self.memory_dimension
        self.max_len_neighbor= max_len_neighbors
        if self.use_history:
            self.his_memory = HistoryMemory(n_nodes=self.n_nodes,
                                memory_dimension=self.memory_dimension,
                                raw_node_features=self.raw_node_features,
                                message_dimension=None,
                                device=device)
        
        self.hierarchy_tree = hierarchy_tree
        
        # output
        self.output_dim = output_dim
        self.layer_class_nums = layer_class_nums
        
        # time encoder
        self.use_time = use_time
        self.time_encoder = TimeEncode(dimension=self.memory_dimension)
        if self.use_time:
            self.time_dim = time_dim
        else:
            self.time_dim = 0
        # memory
        # self.bert_emb_size = bert_emb_size
        self.use_memory = use_memory
        self.memory =None
        self.labeldim = layer_class_nums[self.level - 1]
        if self.use_memory:
            self.memory_dimension = memory_dim
            self.memory_update_at_start = memory_update_at_start
            if self.use_time:
                raw_message_dimension = self.memory_dimension*2 + self.lstm_hidden_size*3 # source_memory+time dimension + source_embedding:concat(text,label,time)
            else:
                raw_message_dimension =  self.memory_dimension*2 + self.lstm_hidden_size *2 # # source_memory+time dimension + source_embedding:concat(text,label)
            message_dimension = self.message_dimension if message_function != "identity" else raw_message_dimension
            self.memory = Memory(n_nodes=self.n_nodes,
                                memory_dimension=self.memory_dimension,
                                input_dimension=message_dimension,
                                message_dimension=message_dimension,
                                device=device)
           

            self.message_aggregator = get_message_aggregator(aggregator_type=message_aggregator_type,
                                                            device=device)
            self.message_function = get_message_function(module_type=message_function,
                                                        raw_message_dimension=raw_message_dimension,
                                                        message_dimension=message_dimension,
                                                        device=device)
            self.memory_updater = get_memory_updater(module_type=memory_updater_type,
                                                    memory=self.memory,
                                               message_dimension=message_dimension,
                                               memory_dimension=self.memory_dimension,
                                               device=device)
            # node embedding
            self.embedding_module_type = node_emb_type
            self.embedding_module = get_embedding_module(self.embedding_module_type, self.nodes_feat_dim, message_dimension, dropout=0.1)
            # self.labeldim = memory_dim
        # GCN
        self.type_text = ["text", "label","labelmem", "ts"]

        if self.com_model_way == 'mean':
            if self.use_time:
                self.fusion = MLLinear(linear_size=[self.com_input_dim + self.labeldim + self.time_dim], output_size=output_dim)
            else:
                self.fusion = MLLinear(linear_size=[self.com_input_dim + self.labeldim], output_size=output_dim)
        elif self.com_model_way == 'lstm':
            if self.use_time:
                self.fusion = MLLinear(linear_size=[self.lstm_hidden_size * 3, self.lstm_hidden_size], output_size=output_dim)
            else:
                self.fusion = MLLinear(linear_size=[self.lstm_hidden_size * 2, self.lstm_hidden_size], output_size=output_dim)
        self.device = device
        
        self.lstm = nn.ModuleDict({t: nn.LSTM(input_size=in_dim, hidden_size=self.lstm_hidden_size, num_layers=1,
                                              batch_first=True) for t, in_dim in
                                   zip(self.type_text, [self.com_input_dim, self.labeldim,self.com_input_dim, self.memory_dimension])})
        self.adap = nn.Linear(2, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.one_hots = torch.eye(layer_class_nums[2])

    def k_hots(self, indexs):
        tensors = []
        for index in indexs:
            res = torch.sum(self.one_hots[index], dim=0)
            tensors.append(res)
        tensors = torch.stack(tensors, dim=0)
        return tensors
    def k_memory_hot(self, batch_labels, com_date, memory):
        label_embeddings = torch.zeros((len(batch_labels), sum(self.layer_class_nums), self.memory_dimension))  
        for i, labels in enumerate(batch_labels): 
            if com_date[i] != 0:
                curr_key = int(str(com_date[i])[0:4]) - 2010
                for idx in labels:
                    if self.level == 3:
                        mem_idx = idx + (self.layer_class_nums[0] + self.layer_class_nums[1])
                    elif self.level == 2:
                        mem_idx = idx + (self.layer_class_nums[0])
                    else:
                        mem_idx = idx
                    if self.use_history:
                        label_embeddings[i, idx, :] = self.his_memory.his_memory[curr_key][mem_idx].clone()
                    else:
                         label_embeddings[i, idx, :] = memory[mem_idx].clone()
            else:
                label_embeddings[i, :, :]  =torch.zeros(100)
        label_embeddings = label_embeddings.to(self.device) 
        return label_embeddings
          
    # 从memory中获取表征
    def k_memory(self, indexs, com_date, memory):
        tensors = []
        i = 0
        for index in indexs:
            res = []
            # res = torch.zeros(self.layer_class_nums[2])
            if com_date[i] != 0:
                curr_key = int(str(com_date[i])[0:4]) - 2010
                for idx in index:
                    mem_idx = idx + (self.layer_class_nums[0] + self.layer_class_nums[1]) # 这块如果是一级+二级数量-1 ，如果分类长度变化需要更改，可以更改为传递
                    if self.use_history:
                        curr_mem = self.his_memory.his_memory[curr_key][mem_idx].clone()
                    else:
                        curr_mem = memory[mem_idx].clone()
                    res.append(curr_mem)
                res = (torch.stack(res, dim=0)).mean(axis=0)
            else:
                res =torch.zeros(100).to(self.device)
            tensors.append(res)
            i += 1
        tensors = torch.stack(tensors, dim=0)
        return tensors
    
    def Embedding(self, ids, clacs_len, word_text_embedding,com_ts,com_date,memory, lstmModel=None, type='text'):
        # print(type, ids)
        if type == 'label':
            if self.use_memory:  
                if True:
                    out = self.k_memory_hot(ids,com_date, memory)
                    outs=lstmModel(out, None, out.shape[0])
                    out = torch.sum(outs, dim=1)
                else:
                    out = self.k_memory(ids,com_date, memory)
                return out.to(self.device)
            else:
                # 不适用层级编码则直接读取k-hot变量
                out = self.k_hots(ids)
                return out.to(self.device)
        if type == 'ts':
            if self.use_time:
                # time diff / max time - min time
                source_time_diffs = ids / 91130.
                # time encoder batch*1*1
                source_time_diffs = source_time_diffs.view(-1,1)
                out = self.time_encoder(source_time_diffs)
                return out.reshape((len(source_time_diffs),-1))  # batchSize * 1
        
        pad_set_embed_seqs = pad_sequence(ids, batch_first=True, padding_value=0)
        pad_set_embed_seqs = pad_set_embed_seqs.to(self.device)
        input_emb = word_text_embedding(pad_set_embed_seqs)[0]  # B * Max_L * F
        if type == 'text' and self.use_lstm_text:
            encoder_outputs = lstmModel(input_emb, torch.tensor(clacs_len))
            lens_unpacked = torch.tensor(clacs_len)
        else:
            embed_input_x_packed = pack_padded_sequence(input_emb, clacs_len, batch_first=True, enforce_sorted=False)
            encoder_outputs, lens_unpacked = pad_packed_sequence(embed_input_x_packed, batch_first=True)

        if type == 'text':
            lens_unpacked = lens_unpacked.unsqueeze(-1).to(self.device)
            out = torch.sum(encoder_outputs, dim=1) / lens_unpacked  # label_feature: N * F
        else:
            out = torch.sum(encoder_outputs, dim=1)
        return encoder_outputs, out

    def fetchFeature(self, textIds, labelIds, textLen, labelLen,com_ts,com_date ,com_num, word_text_embedding,memory, lstmModel, embedding_label=None):
        encoder_text_outputs, text_out = self.Embedding(textIds, textLen, word_text_embedding,com_date, com_ts,memory, lstmModel, type='text')
        text_out = self.dropout(text_out)
        label_out = self.Embedding(labelIds,labelLen,None,com_ts,com_date,memory,
                                                lstmModel,
                                                type='label')
        label_out = self.dropout(label_out)
        if self.use_time:
            ts_out = self.dropout(self.Embedding(com_ts,com_num,None,com_date,None,memory,
                                                embedding_label if embedding_label else self.Embeddings,
                                                type='ts'))
        else:
            ts_out = None
        return {
            "text_enc": encoder_text_outputs,
            "text": text_out,
            "label": label_out,
            "ts":ts_out
        }

    def mean_operation(self, data, lengths):
        pad_set_embed_seqs = pad_sequence(data, batch_first=True, padding_value=0)
        pad_set_label_seqs = pad_set_embed_seqs.to(self.device)

        text_input_x_packed = pack_padded_sequence(pad_set_label_seqs, lengths, batch_first=True, enforce_sorted=False)
        encoder_outputs, lens_unpacked = pad_packed_sequence(text_input_x_packed, batch_first=True)

        lens_unpacked = lens_unpacked.unsqueeze(-1).to(self.device)
        out = torch.sum(encoder_outputs, dim=1) / lens_unpacked  # label_feature: N * F
        return out

    def lstm_operation(self, data, lengths, type_d):
        pad_set_embed_seqs = pad_sequence(data, batch_first=True, padding_value=0)
        pad_set_label_seqs = pad_set_embed_seqs.to(self.device)

        text_input_x_packed = pack_padded_sequence(pad_set_label_seqs, lengths, batch_first=True, enforce_sorted=False)
        outs, (hidden, _) = self.lstm[type_d](text_input_x_packed)
        out = hidden[0]
        return out
    
    def find_parent_nodes(self,index, hierarchy_tree):
        first_level_matrix = hierarchy_tree[0]  
        if self.level == 3:
            second_level_matrix = hierarchy_tree[1]  
            third_level_indices = np.where(second_level_matrix[:, index])[0]          # 找到所有指向给定第三层节点的第二层节点  
            if len(third_level_indices) == 0:  
                print(f"No second-level parents found for the third-level node at index {index}")  
                return [], []  
            # 假设每个第三层节点只有一个第二级父节点（即矩阵中对应列只有一个1）  
            second_level_parent = third_level_indices[0]  
            first_level_indices = np.where(first_level_matrix[:, second_level_parent])[0]
            first_level_parent = first_level_indices[0] 

            return first_level_parent, second_level_parent + first_level_matrix.shape[0] #这里是要和全部分类号的编号对应，一级分类0~8，二级分类8+(1~124),三级分类124+（0~609）
        # 现在找到第二级父节点的第三级父节点（即第一层父节点）  
        else:
            first_level_indices = np.where(first_level_matrix[:, index])[0]
            first_level_parent = first_level_indices[0] 
            return first_level_parent, None
    
    def forward(self,word2vec_emb,bert_text_embeeding,publicationDate,ts, com_text_feats, com_text_lens,com_not_initial,
                lstmModel, lstm_embeddings,) -> torch.Tensor:
        #更改到要在datahelper中处理
        com_patent_num = [sum(item) for item in com_text_lens['com_num']]
                       
        B = len(com_patent_num)
        memory = None
        if self.use_memory:
            # Update memory for all nodes with messages stored in previous batches
            if self.memory_update_at_start:
                #messages 中包含上一个batch中某个分类好的所有message，这里是取最后一个message，作为memory
                memory, last_update = self.get_updated_memory(list(range(self.n_nodes)),self.memory.messages)
            else:
                memory = self.memory.get_memory(list(range(self.n_nodes)))
                last_update = self.memory.last_update
        
        
        init_feats = self.fetchFeature(textIds=com_text_feats['text'], labelIds=com_text_feats['label'],
                                       textLen=com_text_lens['text'], labelLen=com_text_lens['label'],
                                       com_ts = com_text_feats['ts'],com_date = com_text_feats['com_date'],com_num=com_text_lens['com_num'],
                                       word_text_embedding=word2vec_emb, memory=memory,
                                       lstmModel=lstmModel, embedding_label=lstm_embeddings
                                       )
        assert init_feats['text'].shape == torch.Size([sum(com_patent_num), self.com_input_dim])
        # assert init_feats['text'].shape == torch.Size([sum(com_patent_num), self.com_input_dim])
        if self.use_memory:
            assert init_feats['label'].shape == torch.Size([sum(com_patent_num), self.com_input_dim])
        else:
            assert init_feats['label'].shape == torch.Size([sum(com_patent_num), self.labeldim])


        assign_list, readout_feats = dict(), dict()
        assign_list['text'], assign_list['label'] , assign_list['ts']= [], [], []
        start = 0
        to_update_codes = []
        for length in com_patent_num:
            end = start + length
            to_update_codes.append(com_text_feats['label'][start:end])
            assign_list['text'].append(init_feats['text'][start:end])
            assign_list['label'].append(init_feats['label'][start:end])
            if self.use_time:
                assign_list['ts'].append(init_feats['ts'][start:end])
            start = end
        to_update_codes_idx = {}
        labels = []
        for index, item in enumerate(to_update_codes):
            if com_not_initial[index] != 0:
                for label in item:
                    for code in label:
                        code = code.numpy().item()
                        if self.level == 3:
                            sectionCode, classCode = self.find_parent_nodes(code, self.hierarchy_tree)
                            subclassCode = code + self.layer_class_nums[0] + self.layer_class_nums[1]
                            labels.append(sectionCode)
                            labels.append(classCode)
                            labels.append(subclassCode)
                            to_update_codes_idx.setdefault(subclassCode, []).append(index)  
                            to_update_codes_idx.setdefault(classCode, []).append(index)  
                        elif self.level == 2:
                            sectionCode, _= self.find_parent_nodes(code, self.hierarchy_tree)
                            classCode = code + self.layer_class_nums[0]
                            labels.append(sectionCode)
                            labels.append(classCode)
                            to_update_codes_idx.setdefault(subclassCode, []).append(index)
                        else:
                            sectionCode = code
                            labels.append(sectionCode)
                        to_update_codes_idx.setdefault(sectionCode, []).append(index) 
        # 需要更新的三级标签  
        labels = np.array(labels)
            
        if self.com_model_way == 'mean':
            readout_feats['text'] = self.mean_operation(assign_list['text'], com_patent_num)  # label_feature: N * F
            readout_feats['label'] = self.mean_operation(assign_list['label'], com_patent_num)  # label_feature: N * F

        elif self.com_model_way == 'lstm':
            readout_feats['text'] = self.lstm_operation(assign_list['text'], com_patent_num, 'text')
            if self.use_memory:
                readout_feats['label'] = self.lstm_operation(assign_list['label'], com_patent_num, 'labelmem')
            else:
                readout_feats['label'] = self.lstm_operation(assign_list['label'], com_patent_num, 'label')
            if self.use_time:
                readout_feats['ts'] = self.lstm_operation(assign_list['ts'], com_patent_num, 'ts')

        if self.use_time:
            res_emb = self.fusion(torch.cat([readout_feats['text'], readout_feats['label'], readout_feats['ts']], dim=1))
        else:
            res_emb = self.fusion(torch.cat([readout_feats['text'], readout_feats['label']], dim=1))    
        
        assert res_emb.shape == torch.Size([B, self.output_dim])
        
        if len(labels) == 0:
            return res_emb

        if self.use_memory:
            if self.memory_update_at_start:
                # Persist the updates to the memory only for codes in history (since now we have
                # new messages for them)
                self.update_memory(labels, self.memory.messages)
                # assert torch.allclose(memory[labels], self.memory.get_memory(labels), atol=1e-5), \
                # "Something wrong in how the memory was updated"
                # Remove messages for the positives since we have already updated the memory using them
                self.memory.clear_messages(labels)
            if self.use_time:
                source_node_embedding =  torch.cat([readout_feats['text'],readout_feats['label'],readout_feats['ts']],dim =1)
            else:
                source_node_embedding =  torch.cat([readout_feats['text'],readout_feats['label']],dim =1)

            unique_nodes, source_id_to_messages = self.get_raw_messages(labels,source_node_embedding, to_update_codes_idx,ts)
            
            if self.memory_update_at_start:
                self.memory.store_raw_messages(unique_nodes, source_id_to_messages)
            else:
                self.update_memory(unique_nodes, source_id_to_messages)
                
        return res_emb

    def update_memory(self, nodes, messages):
        # Aggregate messages for the same nodes
        unique_nodes, unique_messages, unique_timestamps = \
        self.message_aggregator.aggregate(
            nodes,
            messages)

        if len(unique_nodes) > 0:
            unique_messages = self.message_function.compute_message(unique_messages)

        # Update the memory with the aggregated messages
        self.memory_updater.update_memory(unique_nodes, unique_messages,
                                         timestamps=unique_timestamps)

    def get_updated_memory(self, nodes, messages):
        # Aggregate messages for the same nodes
        unique_nodes, unique_messages, unique_timestamps = \
        self.message_aggregator.aggregate(
            nodes,
            messages)

        if len(unique_nodes) > 0:
            unique_messages = self.message_function.compute_message(unique_messages)

        updated_memory, updated_last_update = self.memory_updater.get_updated_memory(unique_nodes,
                                                                                    unique_messages,
                                                                                    timestamps=unique_timestamps)

        return updated_memory, updated_last_update

    def get_raw_messages(self, source_nodes,source_node_embedding, to_update_codes_idx,ts):
        source_memory = self.raw_node_features[0][source_nodes] + self.memory.get_memory(source_nodes)
        # get idx for current batch
        idx_set = [item for sublist in to_update_codes_idx.values() for item in sublist]
        ts = (torch.tensor(list(ts))[list(idx_set)]).to(self.device)
        source_time_delta =  ts - self.memory.last_update[source_nodes]
        source_time_delta_encoding = self.time_encoder(source_time_delta.unsqueeze(dim=1)).view(len(
        source_nodes), -1)
        source_message = torch.cat([source_memory, source_node_embedding[list(idx_set)],source_time_delta_encoding],
                                dim=1)
        messages = defaultdict(list)
        unique_sources = np.unique(source_nodes)
        for idx, item in enumerate(source_nodes):
            messages[source_nodes[idx]].append((source_message[idx], ts[idx]))
        return unique_sources, messages

if __name__ == '__main__':
    # 定义LSTM模型  
    class LabelEmbeddingLSTM(nn.Module):  
        def __init__(self, input_dim, hidden_dim):  
            super(LabelEmbeddingLSTM, self).__init__()  
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)  
            
        def forward(self, x):  
            # LSTM需要(batch, seq_len, input_size)的输入  
            # 初始化隐藏状态和细胞状态  
            h0 = torch.zeros(1, batch_size, hidden_dim).to(x.device)  # (num_layers * num_directions, batch, hidden_size)  
            c0 = torch.zeros(1, batch_size, hidden_dim).to(x.device)  # (num_layers * num_directions, batch, hidden_size)  
            
            # LSTM的前向传播，得到输出和新的隐藏状态  
            out, _ = self.lstm(x, (h0, c0))  
            
            # 使用最后一个时间步的隐藏状态作为输出  
            # 注意：这里我们取最后一个时间步的输出，其形状为(batch, hidden_dim)  
            last_output = out[:, -1, :]  
            
            # 如果需要，可以将这个形状(batch, hidden_dim)的tensor转换为(batch, 1, hidden_dim)以匹配原始嵌入的维度  
            # last_output = last_output.unsqueeze(1)  
            
            return last_output  
    
        # 假设 batch_size = 8, L = 标签数量, embed_dim = 100  
    batch_size = 8  
    L = 10  # 示例标签数量，实际中可能不同  
    embed_dim = 100  
    hidden_dim = 256  # LSTM隐藏层维度，这里设为100以匹配标签嵌入的维度  
    
    # 随机生成一些标签嵌入数据作为示例  
    label_embeddings = torch.randn(batch_size, L, embed_dim)  
    

    # 实例化模型  
    lstm_model = LabelEmbeddingLSTM(input_dim=embed_dim, hidden_dim=hidden_dim)  
    
    # 将模型移动到适当的设备（如果使用了GPU）  
    # lstm_model.to(device)  # device可以是'cuda'或'cpu'  
    
    # 前向传播得到每个样本的“代表”向量  
    output = lstm_model(label_embeddings)  
    print(output.shape)  # 输出应该是 torch.Size([8, 100])     