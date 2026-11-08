import logging
import warnings
import random
import dgl

import numpy as np
import torch
from collections import defaultdict
import torch.nn.utils.rnn as rnn_utils
import torch.nn as nn
import torch.nn.functional as F

from models.historysequence import HistorySequence
from models.hier_ipc_emb import Hierarchy_Embedding
import transformers

class Embedding(nn.Module):
    """

    """

    def __init__(self, device=None, emb_init=None, padding_idx=0,
                 dropout=0.2):
        super(Embedding, self).__init__()
        self.emb = nn.Embedding.from_pretrained(emb_init, freeze=False)
        self.dropout = nn.Dropout(dropout)
        self.padding_idx = padding_idx
        self.device = device

    def forward(self, inputs):
        input_emb = self.dropout(self.emb(inputs))
        lengths, masks = (inputs != 0).sum(dim=-1), inputs != 0

        return input_emb, lengths, masks

class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, layers_num, dropout, bidirectional=False):
        super(LSTMEncoder, self).__init__()
        self.bidrec = bidirectional
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=layers_num,
                            batch_first=True, bidirectional=bidirectional)
        if self.bidrec:
            self.init_state = nn.Parameter(torch.zeros(2 * 2 * layers_num, 1, hidden_size), requires_grad=False)
        else:
            self.init_state = nn.Parameter(torch.zeros(2 * layers_num, 1, hidden_size), requires_grad=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, lengths, batch_len = 8, **kwargs):
        self.lstm.flatten_parameters()
        if lengths is not None:
            init_state = self.init_state.repeat([1, len(lengths), 1])
            cell_init, hidden_init = init_state[:init_state.size(0) // 2], init_state[init_state.size(0) // 2:]
            packed_inputs = nn.utils.rnn.pack_padded_sequence(inputs, lengths.tolist(), batch_first=True,
                                                          enforce_sorted=False)
       
            if self.bidrec:
                outputs, _ = nn.utils.rnn.pad_packed_sequence(
                    self.lstm(packed_inputs, (hidden_init, cell_init))[0], batch_first=True)
            else:
                outs, (hidden, _) = self.lstm(packed_inputs, (hidden_init, cell_init))
                outputs = outs
        else:
            init_state = self.init_state.repeat([1, batch_len, 1])
            cell_init, hidden_init = init_state[:init_state.size(0) // 2], init_state[init_state.size(0) // 2:]
            if self.bidrec:
                outputs, _ = self.lstm(inputs, (hidden_init, cell_init))
            else:
                outs, (hidden, _) = self.lstm(inputs, (hidden_init, cell_init))
                outputs = outs #hidden[0]
        return self.dropout(outputs)

class TextCNNModel(nn.Module):

    def __init__(self, num_filters, num_classes, hidden_size, device, filter_sizes=(2,3,4), dropout=0.1):
        super(TextCNNModel, self).__init__()
        # kernel 
        self.num_filters = num_filters
        # kernel size
        self.filter_sizes = filter_sizes
        self.num_classes = num_classes
        # same as input embedding dimention size
        self.hidden_size = hidden_size
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters, (k, self.hidden_size)) for k in self.filter_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc_cnn = nn.Linear(self.num_filters * len(self.filter_sizes), self.num_classes)

    def conv_and_pool(self, x, conv):
        '''
        B:batchsize,H:height,W:width,C:Channel
        x:[B,C,H,W]
        '''
        x = conv(x)#  [B,C',H',1]  H'=(H - conv.kernel_size[0] + 2 * conv.padding[0]) // conv.stride + 1,C' = filters 
        x = F.relu(x)#  [B,C',H',1]
        x = x.squeeze(3)# [B,C',H']
        size = x.size(2) # H'
        x = F.max_pool1d(x, size)# [B,C',1]
        x = x.squeeze(2)# [B,C']
        return x
        # x = F.relu(conv(x)).squeeze(3)
        # x = F.max_pool1d(x, x.size(2)).squeeze(2)
        # return x

    def forward(self, encoder_out):
        '''
        encoder_out:(B:batchsize,H:height,W:width)
        '''
        # (B,C:Chanel,H,W) unsqueeze at the second dimension to get 4 dimension input.
        out = encoder_out.unsqueeze(1)
        outputs = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(outputs)
        out = self.fc_cnn(out)
        return outputs, out

class DHPT(torch.nn.Module):
    def __init__(self, 
                 model_name, 
                 bert_emb_size,
                 word_emb_size, 
                 lstm_hidden_size,
                 layers_num, 
                 layer_class_nums, 
                 hierarchy_tree,
                 com_model_way,
                 com_feat_type, 
                 use_hierarchy,  
                 use_vertical,
                 word2vec_emb_init,
                 use_company, 
                 use_history,
                 use_memory,
                 use_time,
                 use_cnn,
                 use_bert, num_filters, filter_sizes,
                 memory_update_at_start,
                 use_lstm_text,
                 message_aggregator_type,
                 n_nodes, 
                 node_emb_type,
                 message_dim, 
                 message_function, 
                 memory_dim,
                 memory_updater_type,
                 raw_node_features,
                 max_len_neighbors,
                 labels_num,
                 level,
                 weight,
                 dropout,
                 device):
        super(DHPT, self).__init__()
        
        self.device = device
        # The output dimension is also the number of class
        self.out_dim = labels_num
        self.weight = weight
        self.fc_hidden_size = lstm_hidden_size #256
        self.text_embedding = word_emb_size #100
        self.hidden_size = lstm_hidden_size
        self.input_dim = word_emb_size
        self.use_cnn = use_cnn
        self.use_bert = use_bert
        self.dropout = dropout
        self.level = level
        # BERT for patent text embedding
        self.bert_dropout = torch.nn.Dropout(dropout)
        if self.use_bert:
            self.text_embedding = bert_emb_size
            self.input_dim = bert_emb_size
            self.bert_emb_size = bert_emb_size
            self.bert_text_embedding = transformers.BertModel.from_pretrained(model_name, return_dict=False)
        else:
            self.word2vec_emb = Embedding(device, word2vec_emb_init, 0, dropout)
            
        if self.use_cnn:
            self.num_filters = num_filters
            self.filter_sizes = filter_sizes
            self.cnn = TextCNNModel(num_filters = self.num_filters, filter_sizes =self.filter_sizes, num_classes=self.out_dim, device = device, hidden_size= self.text_embedding,dropout = dropout)
        
        self.fc1 = torch.nn.Linear(self.text_embedding, self.fc_hidden_size)  
        self.relu = torch.nn.ReLU()  
        self.dropout = torch.nn.Dropout(self.dropout)  
        self.fc2 = torch.nn.Linear(self.fc_hidden_size, labels_num) 
        self.fc3 = torch.nn.Linear(self.text_embedding, labels_num)  

        # LSTM for History
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm = LSTMEncoder(word_emb_size, self.lstm_hidden_size, layers_num, dropout, bidirectional=True)
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # Bi-LSTM
        self.lstm_embedding_label = self.lstm_hidden_size * 2
        self.lstm_embeddings = nn.Embedding(num_embeddings=layer_class_nums[self.level - 1], embedding_dim=self.lstm_embedding_label)

        # use 
        self.use_company = use_company
        self.use_history = use_history
        # word2vec embedding
        self.word2vec_emb = Embedding(device, word2vec_emb_init, 0, 0.2)
        self.use_hierarchy = use_hierarchy
        
        if self.use_company:
            self.histories = HistorySequence(layer_class_nums=layer_class_nums, 
                                             bert_emb_size = self.input_dim, 
                                             lstm_embedding_label = self.lstm_embedding_label, 
                                             lstm_hidden_size=lstm_hidden_size,
                                             use_lstm_text=use_lstm_text, 
                                             com_feat_type=com_feat_type,
                                             com_embedding_text=self.lstm_embedding_label if use_lstm_text else word_emb_size,
                                             com_model_way=com_model_way,
                                             n_nodes=n_nodes, 
                                             message_dim=message_dim,
                                             memory_dim=memory_dim, 
                                             node_emb_type=node_emb_type,
                                             message_function=message_function,
                                             message_aggregator_type=message_aggregator_type,
                                             memory_updater_type=memory_updater_type,
                                             memory_update_at_start = memory_update_at_start,
                                             use_memory=use_memory,
                                             use_time =use_time,
                                             use_history=use_history,
                                             hierarchy_tree=hierarchy_tree,
                                             max_len_neighbors = max_len_neighbors,
                                             raw_node_features = raw_node_features,
                                             time_dim=100,
                                             output_dim=labels_num, 
                                             level = level,
                                             dropout = dropout,
                                             device=device
                                            )    
        
        if use_hierarchy:
            self.local_decoders = Hierarchy_Embedding(input_dim=self.input_dim, hidden_dim=self.input_dim,
                                                      device=device, numbers=layer_class_nums, fuse_layer=1,
                                                      hierarchy_tree=hierarchy_tree,level = level,raw_node_features = raw_node_features)
    def forward(self, bert_text_inputs, bert_text_mask, bert_text_token_type_ids, word_inputs, publicationDate,ts,com_text_feats, com_text_lens,
                com_not_initial):
        # process inputs
        # B, N = inputs.shape
        if self.use_bert:
            B, N = bert_text_inputs.shape
            #BERT embedding: last_hidden, pooler_output stands for a  sentence embedding
            with torch.no_grad():
            # cls
                emb_out, cls_emb_out = self.bert_text_embedding(bert_text_inputs, attention_mask=bert_text_mask, token_type_ids=bert_text_token_type_ids) 
            # use mean vector instead of cls
                sequence_output = emb_out[:, 1:, :]  # 去掉CLS  
                cls_emb_out = torch.mean(sequence_output, dim=1)    # 计算所有词向量的平均值  
            word_mask = bert_text_mask     
        else:
            B, N = word_inputs.shape
            emb_out, length_text, word_mask = self.word2vec_emb(word_inputs)
            assert word_mask.shape == torch.Size([B, N])
            cls_emb_out = emb_out.mean(dim=1)

        local_emb = emb_out
        assert emb_out.shape == torch.Size([B, N, self.text_embedding])
        
        if self.use_cnn:
           _, pred_attn = self.cnn(emb_out)
        else:
            # output_2 = self.bert_dropout(cls_emb_out)
            # pred_attn = self.fc3(output_2)
            x = self.fc1(cls_emb_out)  
            x = self.relu(x)
            x = self.dropout(x)  
            pred_attn = self.fc2(x) 
        
        # WORD2VEC embedding
        dynamic_pred, hierarchy_pred = pred_attn.new_zeros(pred_attn.shape), pred_attn.new_zeros(pred_attn.shape)
       
        if self.use_company and sum(com_not_initial) !=0:
            dynamic_pred_emb = self.histories(self.word2vec_emb, publicationDate,ts,com_text_feats, com_text_lens,com_not_initial, self.lstm,
                                                self.lstm_embeddings)
            dynamic_pred[com_not_initial != 0] = dynamic_pred_emb[com_not_initial != 0]
        
        
        if self.use_hierarchy and self.level > 1:
            hierarchy_pred = self.local_decoders(local_emb, word_mask)   

        return [torch.sigmoid(dynamic_pred * self.weight + pred_attn  + hierarchy_pred)]        

    def compute_update_memory(self, source_nodes, destination_nodes, edge_times):
        """
        param: source_nodes: numpy_array N
        param: destination_nodes: list [[],[],...] N * k
        param: edge_times: numpy_array N
        """
        # step 1: get last update memories (in step 2)

        # step 2: produce message
        unique_src, src_id_to_messages = self.get_company_messages(source_nodes, destination_nodes,
                                                                   edge_times)
        res = self.get_hierarchy_messages(destination_nodes, source_nodes, edge_times)

        # step 3: aggregate message
        unique_messages_src, unique_timestamp_src = self.aggregate_message(unique_src, src_id_to_messages,
                                                                           type=self.type[0])

        # step 4: update memory
        updated_memory_src, update_timestamp_src = self.memory_updater.get_updated_memory(unique_src,
                                                                                          unique_messages_src,
                                                                                          type=self.type[0],
                                                                                          timestamps=unique_timestamp_src)
        update_res = dict()
        update_res[self.type[0]] = [unique_src, updated_memory_src, update_timestamp_src]

        for t, item in res.items():  # todo for k,v in res.items()
            unique_dst, dst_id_to_messages = item  # todo delete dst_nodes
            unique_messages_dst, unique_timestamp_dst = self.aggregate_message(unique_dst, dst_id_to_messages, type=t)
            updated_memory_dst, update_timestamp_dst = self.memory_updater.get_updated_memory(unique_dst,
                                                                                              unique_messages_dst,
                                                                                              type=t,
                                                                                              timestamps=unique_timestamp_dst)
            update_res[t] = [unique_dst, updated_memory_dst, update_timestamp_dst]
        return update_res

    def compute_embedding(self, update_mem):
        # step 5: compute the embeddings
        # {type: nodes, mem, time, time_diff}#

        t_c, t_f = self.type[0], self.type[1]
        company_embedding = self.embedding_module.compute_embedding(memory=update_mem[t_c][1],
                                                                    nodes=update_mem[t_c][0])
        # raw_company_embed = update_mem[t_c][1]
        raw_company_embed = self.memory.get_memory(t_c,range(self.n_nodes[t_c]))
        # print(company_embedding[update_mem[t_c][0]][:25,:10])

        field_embedding = self.embedding_module.compute_embedding(memory=update_mem[t_f][1],
                                                                  nodes=update_mem[t_f][0])

        raw_field_embed = self.memory.get_memory(t_f,range(self.n_nodes[t_f]))

        return company_embedding, field_embedding, \
                raw_company_embed, raw_field_embed,

    def update_self_memory(self, update_mem):
        # nodes, mem, time #
        for t, item in update_mem.items():
            item = update_mem[t]
            self.memory_updater.update_memory(unique_node_ids=item[0], unique_messages=item[1],
                                              type=t, timestamps=item[2])

    def aggregate_message(self, nodes, messages, type):
        # Aggregate messages for the same nodes

        unique_raw_messages = self.message_aggregator.aggregate(nodes, messages)
        unique_messages, unique_timestamp = unique_raw_messages
        if len(nodes) > 0:
            unique_messages = self.message_function[type].compute_message(unique_messages)

        return unique_messages, unique_timestamp

    def padding_nodes(self, data_list):
        """
            :param data_list: shape (batch_users, baskets, item_embed_dim)
            :return:
            """
        warnings.filterwarnings('ignore')
        length = [len(sq) for sq in data_list]
        data_list = [torch.tensor(d) for d in data_list]

        # pad in time dimension
        data = rnn_utils.pad_sequence(data_list, batch_first=True, padding_value=0)
        S = torch.zeros(data.shape)
        for i, l in enumerate(length):
            S[i, l:] = 1

        return data, length, S == 1

    def get_company_messages(self, source_nodes, destination_nodes, edge_times, type="company"):
        """
        source_n : N
        dest_n : N * S(set)
        """
        edge_times = torch.from_numpy(edge_times).to(self.device)

        source_memory = self.memory.get_memory(type, source_nodes)

        dt, length, S = self.padding_nodes(destination_nodes)
        # key = self.memory.get_memory(self.type[1],dt)
        # q = self.memory.get_memory(self.type[1],dt)
        # v = self.memory.get_memory(self.type[1],dt)
        # attn_output, attn_output_weights = self.attn(query=q.transpose(0,1),
        #                                                  key = key.transpose(0,1),
        #                                                  value = v.transpose(0,1),
        #                                              key_padding_mask = S.to(self.device))

        # ------------------- mask mean operation --------------------#
        mask = torch.arange(dt.shape[1]) <= torch.tensor(length).unsqueeze(-1)
        mask = mask.unsqueeze(-1).to(self.device)
        destination_memory = torch.sum(self.memory.get_memory(self.type[1], dt) * mask, dim=1) / mask.sum(1)

        source_time_delta = edge_times - self.memory.last_update[type][source_nodes]

        source_time_delta_encoding = self.time_encoder(source_time_delta.unsqueeze(dim=1))
            # source_time_delta_encoding = self.time_encoder(edge_times.unsqueeze(dim=1))
        source_message = torch.cat([source_memory, destination_memory, source_time_delta_encoding],
                                       dim=1)

        messages = defaultdict(list)
        unique_sources = np.unique(source_nodes)

        for i in range(len(source_nodes)):
            messages[source_nodes[i]].append((source_message[i].unsqueeze(1), edge_times[i]))

        return [unique_sources, messages]

    def get_hierarchy_messages(self, fields_nodes, company_nodes, edge_times):
        # fields_nodes: N_src * S(set) (x: for company i, it has x fields)
        # --------------- layer 1 ------------------ #

        com_nodes, fil_nodes, all_nodes, edge_ts = [], [], [], []

        for i, fn in enumerate(fields_nodes):  # every field has its message
            l = len(fn)
            com_nodes.append(torch.tensor([company_nodes[i]]).expand(l))
            edge_ts.append(torch.tensor([edge_times[i]]).expand(l))
            fil_nodes.append(torch.tensor(fields_nodes[i]))
        all_nodes = torch.cat(fil_nodes, dim=0)
        com_nodes = torch.cat(com_nodes, dim=0)
        edge_ts = torch.cat(edge_ts, dim=0).to(self.device)

        com_memory = self.memory.get_memory(self.type[0], com_nodes)

        result = dict()

        field_memory = self.memory.get_memory(self.type[1], all_nodes)

        source_time_delta = edge_ts - self.memory.last_update[self.type[1]][all_nodes]
        source_time_delta_encoding = self.time_encoder(source_time_delta.unsqueeze(dim=1))
        source_message = torch.cat([field_memory, com_memory, source_time_delta_encoding], dim=1)

        messages = defaultdict(list)
        all_nodes = all_nodes.numpy()
        unique_sources = np.unique(all_nodes)

        for node in range(len(all_nodes)):
            messages[all_nodes[node]].append((source_message[node].unsqueeze(1), edge_ts[node]))

        result[self.type[1]] = (unique_sources, messages)

        return result

    def init_memory(self):
        self.memory.__init_memory__()

    def detach_memory(self):
        self.memory.detach_memory()

    def backup_memory(self):
        return self.memory.backup_memory()

    def restore_memory(self, memory):
        self.memory.restore_memory(memory)

    def sampleNegNode(self, all_nodes, num=10):
        sam_num = min(len(all_nodes) - 1, num)
        nega_list = []
        all_nodes = list(all_nodes)
        for i, node in enumerate(all_nodes):
            nega_list.append(torch.tensor(random.sample(all_nodes[:i] + all_nodes[i + 1:], sam_num)))
        return torch.cat(nega_list)

    def calculate_loss(self, res):
        # dict: {type: unique_src, updated_memory_src, updated_last_update_src, source_time_diffs}#
        regularizer_pos = 0
        for i, t in enumerate(self.type):
            if i > 0 and i + 1 < len(self.type):
                unique_node_i = res[t][0]
                unique_memory_i = res[t][1]
                unique_memory_hig = res[self.type[i + 1]][1]
                reflet_i = self.reflect[i - 1][unique_node_i, 1]
                regularizer_pos += torch.norm(unique_memory_i[unique_node_i] - unique_memory_hig[reflet_i]) ** 2

        return regularizer_pos * self.loss_alpha
