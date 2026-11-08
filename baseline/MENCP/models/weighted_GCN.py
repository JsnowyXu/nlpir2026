import torch.nn as nn
import torch
import dgl
import dgl.function as fn


class weighted_graph_conv(nn.Module):
    """
        Apply graph convolution over an input signal.
    """

    def __init__(self, in_features: int, out_features: int):
        super(weighted_graph_conv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.linear_out = nn.Linear(in_features, out_features, bias=True)

    def forward(self, graph, node_features, edge_weights):
        r"""Compute weighted graph convolution. ----- Input: graph : DGLGraph, batched graph. node_features :
        torch.Tensor, input features for nodes (n_1+n_2+..., in_features) or (n_1+n_2+..., T, in_features)
        edge_weights : torch.Tensor, input weights for edges  (T, n_1^2+n_2^2+..., n^2)

        Output:
        shape: (N, T, out_features)
        """
        graph = graph.local_var()
        node_features = node_features.to('cuda')
        # multi W first to project the features, with bias
        # (N, F) / (N, T, F)
        graph.ndata['n'] = node_features
        # edge_weights, shape (T, N^2)
        # one way: use dgl.function is faster and less requirement of GPU memory
        graph.edata['e'] = edge_weights.t().unsqueeze(dim=-1)  # (E, T, 1)
        graph.update_all(fn.u_mul_e('n', 'e', 'msg'), fn.sum('msg', 'h'))

        # another way: use user defined function, needs more GPU memory
        # graph.edata['e'] = edge_weights.t()
        # graph.update_all(self.gcn_message, self.gcn_reduce)
        readouts = dgl.mean_nodes(graph, 'h')
        node_features = graph.ndata.pop('h')
        output = self.linear(node_features)
        # readouts = self.linear_out(readouts)

        return output, readouts

    @staticmethod
    def gcn_message(edges):
        if edges.src['n'].dim() == 2:
            # (E, T, 1) (E, 1, F),  matmul ->  matmul (E, T, F)
            return {'msg': torch.matmul(edges.data['e'].unsqueeze(dim=-1), edges.src['n'].unsqueeze(dim=1))}

        elif edges.src['n'].dim() == 3:
            # (E, T, 1) (E, T, F),  mul -> (E, T, F)
            return {'msg': torch.mul(edges.data['e'].unsqueeze(dim=-1), edges.src['n'])}

        else:
            raise ValueError(f"wrong shape for edges.src['n'], the length of shape is {edges.src['n'].dim()}")

    @staticmethod
    def gcn_reduce(nodes):
        # propagate, the first dimension is nodes num in a batch
        # h, tensor, shape, (N, neighbors, T, F) -> (N, T, F)
        return {'h': torch.sum(nodes.mailbox['msg'], 1)}


class weighted_GCN(nn.Module):
    def __init__(self, in_features: dict, hidden_sizes: dict, out_features: dict, type_feat: list):
        super(weighted_GCN, self).__init__()
        self.type_feat = type_feat
        gcns, relus, bns, linear = nn.ModuleDict(
            {t: nn.ModuleList() for t in type_feat}), nn.ModuleDict(
            {t: nn.ModuleList() for t in type_feat}), nn.ModuleDict(
            {t: nn.ModuleList() for t in type_feat}), nn.ModuleDict()
        transform_q, transform_k, transform_v, transform_F = nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        # layers for hidden_size
        self.lay_nums = len(hidden_sizes[type_feat[0]])
        for i, t in enumerate(type_feat):
            if t == 'ts':
                continue
            input_size = in_features[t]
            for hidden_size in hidden_sizes[t]:
                gcns[t].append(weighted_graph_conv(input_size, hidden_size))
                relus[t].append(nn.ReLU())
                bns[t].append(nn.BatchNorm1d(hidden_size))
                input_size = hidden_size
                if i == 0:
                    transform_q.append(nn.Linear(hidden_size, hidden_size))
                    transform_F.append(nn.Linear(hidden_size, hidden_size))
                    transform_k.append(nn.Linear(hidden_size, hidden_size))
                    transform_v.append(nn.Linear(hidden_size, hidden_size))
            # output layer
            gcns[t].append(weighted_graph_conv(hidden_sizes[t][-1], out_features[t]))
            relus[t].append(nn.ReLU())
            bns[t].append(nn.BatchNorm1d(out_features[t]))
            linear[t] = nn.Linear(hidden_sizes[t][-1], out_features[t])
        self.gcns, self.relus, self.bns, self.transq, self.transk, self.transv, self.transf, self.linear = \
            gcns, relus, bns, transform_q, transform_k, transform_v, transform_F, linear
        self.dropout = nn.Dropout(0.5)

    def transform_attn(self, input_text, input_label, layer):
        input_text_f = self.transf[layer](input_text)
        input_label_f = self.transf[layer](input_label)
        key_d = torch.stack([input_text_f, input_label_f], dim=1)  # N * 2 * F
        data_q = self.transq[layer](input_text_f).unsqueeze(dim=1)  # N * 1 * F
        data_k = self.transk[layer](key_d)  # N * 2 * F
        data_v = self.transv[layer](key_d)  # N * 2 * F
        attn = torch.softmax(torch.sum(data_q * data_k, dim=-1), dim=-1).unsqueeze(dim=1)  # N * 1 * 2
        # print(attn)
        out = self.dropout(torch.matmul(attn, data_v).squeeze(dim=1))
        return out
        # data_q = self.dropout(self.transq[layer](input_text))
        # data_k = self.dropout(self.transk[layer](input_label))
        # data_v = self.dropout(self.transv[layer](input_label))
        # attn = torch.softmax(torch.matmul(data_q, data_k.transpose(0, 1)) / math.sqrt(data_q.shape[1]), dim=1)
        # return torch.matmul(attn, data_v)

    def forward(self, graph: dgl.DGLGraph, node_features: dict, edges_weight: torch.Tensor):
        """
        :param graph: a graph
        :param node_features: shape (n_1+n_2+..., n_features)
        :param edges_weight: shape (T, n_1^2+n_2^2+...)
        :return:
        """
        h = node_features
        readouts = dict()
        for layer in range(self.lay_nums):
            for t in self.type_feat:
                if t == 'ts':
                    continue
                gcn, relu, bn = self.gcns[t][layer], self.relus[t][layer], self.bns[t][layer]
                h[t], readout_t = gcn(graph, h[t], edges_weight)
                h[t] = relu(bn(self.dropout(h[t])))
                readouts[t] = readout_t

            # attention_function
            # new_h = dict()
            # new_h[self.type_feat[0]] = self.transform_attn(h[self.type_feat[0]], h[self.type_feat[1]], layer)
            # new_h[self.type_feat[1]] = self.transform_attn(h[self.type_feat[1]], h[self.type_feat[0]], layer)
            # h = new_h
        readouts = {t: self.dropout(self.linear[t](readouts[t])) for t in self.type_feat}
        return h, readouts


class stacked_weighted_GCN_blocks(nn.ModuleList):
    def __init__(self, *args, **kwargs):
        super(stacked_weighted_GCN_blocks, self).__init__(*args, **kwargs)

    def forward(self, *input):
        g, nodes_feature, edge_weights = input
        h = nodes_feature
        for module in self:
            h = module(g, h, edge_weights)
        return h


def get_edges_weight(patents, max_window=30):
    edges_weight_dict = torch.zeros(len(patents), len(patents))
    for i in range(len(patents)):
        edges_weight_dict[i, i] += 1.
        for j in range(i + 1, min(len(patents), i + 1 + max_window)):
            edges_weight_dict[i, j] += 1. / (j - i)
            edges_weight_dict[j, i] += 1. / (j - i)
    return edges_weight_dict


if __name__ == '__main__':
    nodes_list = [[10, 20, 30, 40, 27, 18],
                  [90, 28, 34, 100],
                  [90, 233, 789],
                  [2, 3, 1, 3, 5],
                  [6, 3, 3, 3, 5, 7]]
    items = []
    for nodes in nodes_list:
        project_nodes = torch.tensor(list(range(len(nodes))))
        src = torch.stack([project_nodes for _ in range(project_nodes.shape[0])], dim=1).flatten().tolist()
        dst = torch.stack([project_nodes for _ in range(project_nodes.shape[0])], dim=0).flatten().tolist()
        g = dgl.graph((src, dst), num_nodes=project_nodes.shape[0])
        edges_weight = get_edges_weight(nodes)

        items.append((g, edges_weight, torch.tensor(nodes)))
    input = []
    for idx, item in enumerate(zip(*items)):
        print(type(item[0]))
        if isinstance(item[0], dgl.DGLGraph):
            print(item)
            input.append(dgl.batch(item))
        elif isinstance(item[0], torch.Tensor):
            if idx == 1:
                # pad edges_weight sequence in time dimension batch, (T, N*N)
                # (T_max, N*N)
                max_length = max([data.shape[0] for data in item])
                edges_weight, lengths = list(), list()
                for data in item:
                    edges_weight.append(data.flatten())
                    lengths.append(data.shape[0])
                # (T_max, N_1*N_1 + N_2*N_2 + ... + N_b*N_b)
                input.append(torch.cat(edges_weight))
                input.append(torch.tensor(lengths))
            else:
                # nodes_feature -> (N_1 + N_2, .. + N_b, item_embedding) or nodes -> (N_1 + N_2, .. + N_b, )
                input.append(torch.cat(item, dim=0))
    # print(input)
    model = weighted_GCN(in_features=3, hidden_sizes=[10, 10], out_features=4)
    input_dt = torch.cat([torch.rand(len(node), 3) for node in nodes_list], dim=0)
    print(len(input))
    print(input[0])
    print(input[3])
    output, h = model(input[0], input_dt, input[1])
    print(h.shape)
    print(output[:6, :].mean(dim=0))
    print(h)
