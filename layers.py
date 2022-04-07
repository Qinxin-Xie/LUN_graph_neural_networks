import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


class GraphConvolution(Module):

    def __init__(self, input_features, output_features, dropout=0., activate_function=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = input_features
        self.out_features = output_features
        self.dropout = dropout
        self.acf = activate_function
        self.weight = Parameter(torch.FloatTensor(input_features, output_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj_matrix):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj_matrix, support)
        output = self.acf(output)

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class DeepGraphConvolution(Module):

    def __init__(self, input_features, output_features, dropout=0., activate_function=F.relu):
        super(DeepGraphConvolution, self).__init__()
        self.in_features = input_features
        self.out_features = output_features
        self.dropout = dropout
        self.acf = activate_function
        self.weight1 = Parameter(torch.FloatTensor(input_features, input_features))
        self.weight2 = Parameter(torch.FloatTensor(input_features, output_features))
        self.weight3 = Parameter(torch.FloatTensor(output_features, output_features))
        self.reset_parameters()

    def reset_parameters(self):
        for weight in [self.weight1, self.weight2, self.weight3]:
            torch.nn.init.xavier_uniform_(weight)

    def forward(self, input, adj_matrix):
        input = F.dropout(input, self.dropout, self.training)
        support1 = torch.mm(input, self.weight1)
        output1 = torch.spmm(adj_matrix, support1)
        output1 = self.acf(output1)
        
        support2 = torch.mm(output1, self.weight2)
        output2 = torch.spmm(adj_matrix, support2)
        output2 = self.acf(output2)
        
        support3 = torch.mm(output2, self.weight3)
        output3 = torch.spmm(adj_matrix, support3)
        output3 = self.acf(output3)
        
        return output3

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
               
class GraphAttentionLayer(nn.Module):

    def __init__(self, input_features, output_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = input_features
        self.out_features = output_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(input_features, output_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*output_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj_matrix):
        h = torch.mm(input, self.W)
        # print(h.size())
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj_matrix > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime), attention
        else:
            return h_prime, attention

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
