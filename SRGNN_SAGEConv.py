import os
import time
import fire
import joblib
import random
from os.path import abspath, dirname, join as pjoin

import torch
from torch import nn
from easydict import EasyDict
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch.nn.utils.rnn import pad_sequence

import sys
sys.path.append(abspath(dirname(dirname(__file__))))

from base import Model
from loader import SessionDataset, SessionDataLoader
from utils import get_fallback_items

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class SRGNN(Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class SessionGraph(Module):
    def __init__(self, hidden_channels=512, batchSize= 512, step=1,  #nonhybrid=TRUE,n_node=23691):
        super(SessionGraph, self).__init__()
        self.hidden_size = hidden_channels
        self.n_node = n_node
        self.batch_size = batchSize
        #self.nonhybrid = nonhybrid
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.gnn = SRGNN(self.hidden_size, step=1)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr= 2e-3, weight_decay=1e-6)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden, mask):
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, ht], 1))
        b = self.embedding.weight[1:]  # n_nodes x latent_size
        scores = torch.matmul(a, b.transpose(1, 0))
        return scores

    def forward(self, inputs, A):
        hidden = self.embedding(inputs)
        hidden = self.gnn(A, hidden)
        return hidden



class GNN(Model, Module):
    def __init__(self, idmap, features, edge_index_2, hidden_channels=512, out_channels=256, dropout=0.1):
        super(GNN, self).__init__()            
        self.idmap = idmap
        self.idmap_inv = {v: k for k, v in idmap.items()}
        self.n_items = len(self.idmap)
        self.x = torch.nn.Parameter(features, requires_grad=False)
        self.edge_index = torch.nn.Parameter(edge_index_2, requires_grad=False)
        self.convs = nn.ModuleList([
            SAGEConv(-1, hidden_channels),
            SAGEConv(-1, out_channels)
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_channels), 
            nn.BatchNorm1d(out_channels),
        ])
        self.drop = nn.Dropout(dropout)
        self.cross_entropy = nn.CrossEntropyLoss()
        
        #down here is SRGNN
        self.hidden_size = hidden_channels
        self.n_node = n_node
        self.batch_size = batchSize
        #self.nonhybrid = nonhybrid
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.gnn = SRGNN(self.hidden_size, step=1)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        #self.loss_function = nn.CrossEntropyLoss()
        #self.optimizer = torch.optim.Adam(self.parameters(), lr= 2e-3, weight_decay=1e-6)
        #self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

    def _forward(self):
        x = self.x
        for conv, bn in zip(self.convs, self.bns):
            # h = self.drop(conv(x, self.edge_index, self.edge_weight).tanh())
            h = self.drop(conv(x, self.edge_index).tanh())
            x = bn(h)
        return x
    
    def session_emb(self, batch, E=None):
        if E is None:
            E = self._forward()
        E = torch.vstack((E, E.mean(0)))
        views = pad_sequence(batch.views, batch_first=True, padding_value=self.n_items).to(self.device)
        Eviews = F.embedding(views, E, padding_idx=self.n_items)
        mask = views != self.n_items
        return (Eviews * mask.unsqueeze(-1)).sum(dim=1) / batch.extra.seq_lens.unsqueeze(dim=-1).to(self.device)
    
    def get_embeddings(self, batch):
        E = self._forward()
        ret = {'sessions': self.session_emb(batch, E), 'positives': F.embedding(batch.purchases.to(self.device), E)}
        if batch.extra.get('negatives', None) is not None:
            ret.update({'negatives': F.embedding(batch.extra.negatives.to(self.device), E)})
        return EasyDict(ret)
        
    # down is SRGNN
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden, mask):
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, ht], 1))
        b = self.embedding.weight[1:]  # n_nodes x latent_size
        scores = torch.matmul(a, b.transpose(1, 0))
        return scores

    '''def forward(self, inputs, A):
        hidden = self.embedding(inputs)
        hidden = self.gnn(A, hidden)
        return hidden
       ''' 
    def forward(self, batch, mask=True, normalize_sessions=False, inputs, A):
        E = self._forward()
        Esess = self.session_emb(batch, E)
        if normalize_sessions:
            Esess = F.normalize(Esess)
        logit = Esess @ F.normalize(E.T)
        # logit = Esess @ E.T
        if mask:
            logit[batch.extra.histories] = -10000.0
            
        hidden = self.embedding(inputs)
        hidden = self.gnn(A, hidden)
        
        return logit+hidden/2


def run(
    epochs=50, batch_size=512, device='cuda', seed=2022,
    num_workers=0, num_negatives=100, pin_memory=False, persistent_workers=False,
    save_fname='save/gnn.pt', submit=False, **kwargs
):
    print(locals())
    sleep_seconds = random.randint(1, 10)
    print(f'sleep {sleep_seconds} before start')
    time.sleep(sleep_seconds)
    dir_name = '/content/drive/MyDrive/processed_submit'
    idmap, fidmap = joblib.load(f'{dir_name}/indices')[:2]
    aug = '_aug' if kwargs.get('augmentation', False) else ''
    df_train = joblib.load(f'{dir_name}/df_train{aug}')
    df_te = joblib.load(f'{dir_name}/df_val')
    features = torch.FloatTensor(joblib.load(f'{dir_name}/features'))
    edge_index_2 = joblib.load(f'{dir_name}/edge_index_v2.0')[0]
    edge_index_1 = joblib.load(f'{dir_name}/edge_index_v1.1')[0]
    model = GNN(idmap, features, edge_index_2).to(device)
    model.set_fork_logic(df_train, df_te, joblib.load(f'{dir_name}/features'))
    ds_tr = SessionDataset(df_train, idmap=idmap, fidmap=fidmap, num_negatives=num_negatives)
    dl_tr = SessionDataLoader(
        ds_tr, batch_size=batch_size, num_negatives=num_negatives, idmap=idmap, fidmap=fidmap,
        num_workers=num_workers, persistent_workers=persistent_workers, pin_memory=pin_memory,
        shuffle=kwargs.get('shuffle', False)
    )
    ds_te = SessionDataset(df_te, idmap=idmap, fidmap=fidmap, num_negatives=num_negatives)
    dl_te = SessionDataLoader(ds_te, batch_size=batch_size, num_negatives=num_negatives, idmap=idmap, fidmap=fidmap)
    if os.path.exists(save_fname):
        model.load_state_dict(torch.load(save_fname))
        hit, mrr = model.validate(dl_te, **kwargs)
        print(f'HIT: {hit:.6f}, MRR: {mrr:.8f}')
        return
    optimizer = torch.optim.Adam(model.parameters(),  lr=2e-3, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, threshold=1e-6, verbose=1)
    model.fit(dl_tr, dl_te, optimizer, scheduler, epochs=epochs, save_fname=save_fname, seed=seed)


if __name__ == '__main__':
    fire.Fire(run)
