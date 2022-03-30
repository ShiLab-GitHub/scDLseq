import torch
from torch import nn, optim
import torch.nn.functional as F
from lsoftmax import LSoftmaxLinear
import math


class BiLSTM_Attention_L(nn.Module):

    def __init__(self, vocab_size, embedding_dim, output_size, hidden_dim, n_layers, margin, device):
        super(BiLSTM_Attention_L, self).__init__()
        self.margin = margin
        self.device = device
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.output_size = output_size
        self.embedding.weight.requires_grad = True
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=True, dropout=0.1)
        
        self.dropout = nn.Dropout(0.1)
        self.w_omega = nn.Parameter(torch.Tensor(hidden_dim * 2, hidden_dim * 2))
        self.u_omega = nn.Parameter(torch.Tensor(hidden_dim * 2, 1))
        self.lsoftmax_linear = LSoftmaxLinear(
            input_features=hidden_dim * 2, output_features=output_size, margin=margin, device=self.device)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.lsoftmax_linear.reset_parameters()
    def attention_net(self, x, query, mask=None):      
        d_k = query.size(-1)                                              
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)  
        p_attn = F.softmax(scores, dim = -1)                              
        context = torch.matmul(p_attn, x).sum(1)
        return context, p_attn


    def forward(self, x, target=None):
        embedding = self.dropout(self.embedding(x))       
        output, (final_hidden_state, final_cell_state) = self.rnn(embedding)
        output = output.permute(1, 0, 2)
        query = self.dropout(output)
        attn_output, attention = self.attention_net(output, query)       
        logit = self.lsoftmax_linear(input=attn_output, target=target)
        return logit, attention


class BiLSTM_Attention_L_vis(nn.Module):

    def __init__(self, vocab_size, embedding_dim, output_size, hidden_dim, n_layers, margin, device):
        super(BiLSTM_Attention_L_vis, self).__init__()
        self.margin = margin
        self.device = device
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.output_size = output_size
        self.embedding.weight.requires_grad = True
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim * 2, 2)
        self.dropout = nn.Dropout(0.1)
        self.w_omega = nn.Parameter(torch.Tensor(hidden_dim * 2, hidden_dim * 2))
        self.u_omega = nn.Parameter(torch.Tensor(hidden_dim * 2, 1))
        self.lsoftmax_linear = LSoftmaxLinear(
            input_features=2, output_features=output_size, margin=margin, device=self.device)
        self.reset_parameters()
    def reset_parameters(self):
        self.lsoftmax_linear.reset_parameters()
    def attention_net(self, x, query, mask=None):
        d_k = query.size(-1)                
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)  
        p_attn = F.softmax(scores, dim = -1)                              
        context = torch.matmul(p_attn, x).sum(1)
        return context, p_attn


    def forward(self, x, target=None):
        embedding = self.dropout(self.embedding(x))
        output, (final_hidden_state, final_cell_state) = self.rnn(embedding)
        output = output.permute(1, 0, 2)
        query = self.dropout(output)
        attn_output, attention = self.attention_net(output, query)       
        
        fc_out = self.fc(attn_output)
        logit = self.lsoftmax_linear(input=fc_out, target=target)
        return logit,fc_out

