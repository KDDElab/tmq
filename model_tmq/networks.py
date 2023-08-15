"""
This script implements an outlier interpretation method of the following paper:
"Beyond Outlier Detection: Outlier Interpretation by Attention-Guided Triplet Deviation Network". in WWW'21.
@ Author: Hongzuo Xu
@ email: hongzuo.xu@gmail.com or leogarcia@126.com or xuhongzuo13@nudt.edu.cn
"""


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class TMQnet(nn.Module):
    def __init__(self, attn_net, attn_net_local, attn_positive, attn_negative, attn_namon, n_feature, n_linear, margin, alpha1, alpha2):
        super(TMQnet, self).__init__()
        self.attn_net = attn_net
        self.attn_net_local = attn_net_local
        self.attn_net_positive = attn_positive
        self.attn_net_negative = attn_negative
        self.attn_net_namon = attn_namon
        self.margin = margin
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        # 线性变换函数,n_feature是输入样本的大小,n_linear是输出样本的大小
        self.linear = torch.nn.Linear(n_feature, n_linear, bias=False) 
        self.attnplus_net =  AttentionNetplus(in_feature=n_linear * 4, n_hidden=2*n_linear, out_feature=n_linear)
        self.attnplus_net_local =  AttentionNetplus(in_feature=n_linear * 3, n_hidden=int(1.5)*n_linear, out_feature=n_linear)
        self.myloss = MyLoss(alpha1=self.alpha1,alpha2=self.alpha2,margin=self.margin)

    def forward(self, anchor, positive, negative, nanom):
        anchor = self.linear(anchor)
        positive = self.linear(positive)
        negative = self.linear(negative)
        nanom = self.linear(nanom)

        # 全局global multi-head Attention
        query = torch.cat([negative, anchor, positive, nanom], dim=1)
        q = []
        q.append(query)
        q = torch.stack(q, dim=0)
        Multihead_attn_net = self.attn_net(q, q, q)
        attn_global = self.attnplus_net(Multihead_attn_net)

        # 局部local multi-head Attention (o,o2,m)
        key_local = torch.cat([negative, nanom, anchor], dim=1)
        k = []
        k.append(key_local)
        k = torch.stack(k, dim=0)
        Multihead_attn_net_local1 = self.attn_net_local(k, k, k)
        attn_anchor = self.attnplus_net_local(Multihead_attn_net_local1)


        # 局部local multi-head Attention (o,o2,n)
        value_local = torch.cat([negative, nanom, positive], dim=1)
        v = []
        v.append(value_local)
        v = torch.stack(v, dim=0)
        Multihead_attn_net_local2 = self.attn_net_positive(v, v, v)
        attn_positive = self.attnplus_net_local(Multihead_attn_net_local2)

        # 局部local multi-head Attention (m,n,o)
        o_local = torch.cat([anchor, positive, negative], dim=1)
        o = []
        o.append(o_local)
        o = torch.stack(o, dim=0)
        Multihead_attn_net_negative = self.attn_net_negative(o, o, o)
        attn_negative = self.attnplus_net_local(Multihead_attn_net_negative)

        # 局部local multi-head Attention (m,n,o2)
        o2_local = torch.cat([anchor, positive, negative], dim=1)
        o2 = []
        o2.append(o2_local)
        o2 = torch.stack(o2, dim=0)
        Multihead_attn_net_namon = self.attn_net_namon(o2, o2, o2)
        attn_namon = self.attnplus_net_local(Multihead_attn_net_namon)


        attn_midd = attn_global + attn_anchor + attn_positive + attn_negative + attn_namon
        
        # attn_midd = attn_global
        _min = torch.unsqueeze(torch.min(attn_midd, dim=1)[0], 0).t()
        _max = torch.unsqueeze(torch.max(attn_midd, dim=1)[0], 0).t()
        attn = (attn_midd - _min) / (_max - _min)

        embedded_n = negative * attn
        embedded_a = anchor * attn
        embedded_p = positive * attn
        embedded_nn = nanom * attn

        embedded_n_dff = (1 - attn) * negative
        embedded_a_dff = (1 - attn) * anchor
        embedded_p_dff = (1 - attn) * positive
        embedded_nn_dff = (1 - attn) * nanom

        dis1 = F.pairwise_distance(embedded_n_dff, embedded_a_dff)
        dis2 = F.pairwise_distance(embedded_p_dff, embedded_a_dff)

        dis3 = F.pairwise_distance(embedded_n_dff, embedded_nn_dff)

        dis = torch.abs(dis1 - dis2)+torch.abs(dis1-dis3)+torch.abs(dis3-dis2)

        #return embedded_a, embedded_p, embedded_n, embedded_nn, attn, dis
        return self.myloss(embedded_a, embedded_p, embedded_n, embedded_nn, dis), attn, dis
    

    def get_lnr(self, x):
        return self.linear(x)


class AttentionNet(nn.Module):
    def __init__(self, in_feature, n_hidden, out_feature):
        super(AttentionNet, self).__init__()
        self.hidden = torch.nn.Linear(in_feature, n_hidden)
        self.out = torch.nn.Linear(n_hidden, out_feature)

    def forward(self, x): 
        x = torch.relu(self.hidden(x))
        x = self.out(x)
        _min = torch.unsqueeze(torch.min(x, dim=1)[0], 0).t()
        _max = torch.unsqueeze(torch.max(x, dim=1)[0], 0).t()
        x = (x - _min) / (_max - _min)
        return x


class AttentionNetplus(nn.Module):
    def __init__(self, in_feature, n_hidden, out_feature):
        super(AttentionNetplus, self).__init__()
        self.hidden = torch.nn.Linear(in_feature, n_hidden)
        self.out = torch.nn.Linear(in_feature, out_feature)

    def forward(self, x):                                                                         
        x = self.out(x)
        _min = torch.unsqueeze(torch.min(x, dim=1)[0], 0).t()
        _max = torch.unsqueeze(torch.max(x, dim=1)[0], 0).t()
        x = (x - _min) / (_max - _min)
        return x



class MyMultiheadAttention(nn.Module):
    def __init__(self, in_feature, num_heads=4, dropout=0., bias=False):
        super(MyMultiheadAttention, self).__init__()
        self.embed_dim = in_feature
        self.head_dim = in_feature // num_heads
        self.kdim = self.head_dim
        self.vdim = self.head_dim
        self.num_heads = num_heads
        self.dropout = dropout
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim 除以 num_heads必须为整数"
        self.q_proj_weight = nn.Parameter(torch.Tensor(in_feature, in_feature))
        self.k_proj_weight = nn.Parameter(torch.Tensor(in_feature, in_feature))  
        self.v_proj_weight = nn.Parameter(torch.Tensor(in_feature, in_feature))
        self.out_proj = nn.Linear(in_feature, in_feature, bias=bias)

        nn.init.xavier_uniform_(self.q_proj_weight)
        nn.init.xavier_uniform_(self.k_proj_weight)
        nn.init.xavier_uniform_(self.v_proj_weight)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        x = self.multi_head_attention_forward(query, key, value, self.num_heads,
                                            self.dropout, self.out_proj.weight, self.out_proj.bias,
                                            training=self.training,
                                            key_padding_mask=key_padding_mask,
                                            q_proj_weight=self.q_proj_weight,
                                            k_proj_weight=self.k_proj_weight,
                                            v_proj_weight=self.v_proj_weight,
                                            attn_mask=attn_mask)

        return x

    def multi_head_attention_forward(self, query, key, value, num_heads, dropout_p, out_proj_weight, out_proj_bias, training=True, key_padding_mask=None,
                                     q_proj_weight=None, k_proj_weight=None, v_proj_weight=None, attn_mask=None):
        q = F.linear(query, q_proj_weight)
        k = F.linear(key, k_proj_weight)
        v = F.linear(value, v_proj_weight)
        bsz, tgt_len, embed_dim = query.size()
        src_len = key.size(1)
        head_dim = embed_dim // num_heads
        scaling = float(head_dim) ** -0.5
        q = q * scaling

        q = q.contiguous().view(bsz * num_heads, tgt_len, head_dim)
        k = k.contiguous().view(bsz * num_heads, tgt_len, head_dim)
        v = v.contiguous().view(bsz * num_heads, tgt_len, head_dim)
        attn_output_weights = torch.bmm(q, k.transpose(1,2))

        if attn_mask is not None:
            attn_output_weights += attn_mask
        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len,src_len) 

        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)
        attn_output = torch.bmm(attn_output_weights, v)
       
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, embed_dim)
        return attn_output

class MyMultiheadAttention_local(nn.Module):
    def __init__(self, in_feature, num_heads=3, dropout=0., bias=False):
        super(MyMultiheadAttention_local, self).__init__()
        self.embed_dim = in_feature
        self.head_dim = in_feature // num_heads
        #print("self.head_dim", self.head_dim)
        self.kdim = self.head_dim
        self.vdim = self.head_dim
        self.num_heads = num_heads
        self.dropout = dropout
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim 除以 num_heads必须为整数"
        self.q_proj_weight = nn.Parameter(torch.Tensor(in_feature, in_feature))
        self.k_proj_weight = nn.Parameter(torch.Tensor(in_feature, in_feature))  
        self.v_proj_weight = nn.Parameter(torch.Tensor(in_feature, in_feature))
        self.out_proj = nn.Linear(in_feature, in_feature, bias=bias)

        nn.init.xavier_uniform_(self.q_proj_weight)
        nn.init.xavier_uniform_(self.k_proj_weight)
        nn.init.xavier_uniform_(self.v_proj_weight)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        x = self.multi_head_attention_forward(query, key, value, self.num_heads,
                                            self.dropout, self.out_proj.weight, self.out_proj.bias,
                                            training=self.training,
                                            key_padding_mask=key_padding_mask,
                                            q_proj_weight=self.q_proj_weight,
                                            k_proj_weight=self.k_proj_weight,
                                            v_proj_weight=self.v_proj_weight,
                                            attn_mask=attn_mask)

        return x

    def multi_head_attention_forward(self, query, key, value, num_heads, dropout_p, out_proj_weight, out_proj_bias, training=True, key_padding_mask=None,
                                     q_proj_weight=None, k_proj_weight=None, v_proj_weight=None, attn_mask=None):
        q = F.linear(query, q_proj_weight)
        k = F.linear(key, k_proj_weight)
        v = F.linear(value, v_proj_weight)
        bsz, tgt_len, embed_dim = query.size()
        src_len = key.size(1)
        head_dim = embed_dim // num_heads
        scaling = float(head_dim) ** -0.5
        q = q * scaling

        q = q.contiguous().view(bsz * num_heads, tgt_len, head_dim)
        k = k.contiguous().view(bsz * num_heads, tgt_len, head_dim)
        v = v.contiguous().view(bsz * num_heads, tgt_len, head_dim)
        attn_output_weights = torch.bmm(q, k.transpose(1,2))

        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)
        attn_output = torch.bmm(attn_output_weights, v)
       
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, embed_dim)
        return attn_output


class MyLoss(nn.Module):
    def __init__(self, alpha1, alpha2, margin):
        super(MyLoss, self).__init__()
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.margin = margin
        self.criterion_tml = torch.nn.TripletMarginLoss(margin=margin, p=2)
        self.sigma_weight = nn.Parameter(torch.Tensor(2, 1))
        nn.init.xavier_uniform_(self.sigma_weight)
        return
    
    def Quadrupletloss(self, embed_anchor, embed_pos, embed_neg, embed_nn, swap=False, reduction="mean", p=2, eps=1e-6):
        def _np_distance(input1, input2, p, eps):

            np_pnorm = np.power(torch.abs((input1.cpu().detach() - input2.cpu().detach() + eps)), p)
            np_pnorm = np.power(torch.sum(np_pnorm, -1), 1.0 / p)
            return np_pnorm
        
        dist_pos = _np_distance(embed_anchor, embed_pos, p, eps)
        dist_neg = _np_distance(embed_neg, embed_nn, p, eps)

        if swap:
            dist_swap = _np_distance(embed_anchor, embed_pos, p, eps)
            dist_neg = np.minimum(dist_neg, dist_swap)
        output = np.maximum(self.margin + dist_pos - dist_neg, 0)

        if reduction == "mean":
            return torch.mean(output)
        elif reduction == "sum":
            return torch.sum(output)
        else:
            return output

    

    def forward(self, embed_anchor, embed_pos, embed_neg, embed_nn, dis):
        loss_tml = (1/(2*self.sigma_weight[0]**2))*self.Quadrupletloss(embed_anchor, embed_pos, embed_neg, embed_nn)+(1/(2*self.sigma_weight[0]**2))*self.criterion_tml(embed_anchor, embed_pos, embed_neg)+(1/(2*self.sigma_weight[1]**2))*self.criterion_tml(embed_anchor, embed_pos, embed_nn)
        +torch.log(self.sigma_weight[0]*self.sigma_weight[1])
        # loss_tml = self.Quadrupletloss(embed_anchor, embed_pos, embed_neg, embed_nn)+self.criterion_tml(embed_anchor, embed_pos, embed_neg)+self.criterion_tml(embed_anchor, embed_neg, embed_nn)
        loss_dis = torch.mean(dis)
        loss = self.alpha1 * loss_tml + self.alpha2 * loss_dis
        return loss



class MyLoss_Q(nn.Module):
    def __init__(self, alpha1, alpha2, margin):
        super(MyLoss_Q, self).__init__()
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.margin = margin
        return
    
    def Quadrupletloss(self, embed_anchor, embed_pos, embed_neg, embed_nn, swap=False, reduction="mean", p=2, eps=1e-6):
        def _np_distance(input1, input2, p, eps):
            np_pnorm = np.power(torch.abs((input1.cpu().detach() - input2.cpu().detach() + eps)), p)
            np_pnorm = np.power(torch.sum(np_pnorm, -1), 1.0 / p)
            return np_pnorm
        
        dist_pos = _np_distance(embed_anchor, embed_pos, p, eps)
        dist_neg = _np_distance(embed_neg, embed_nn, p, eps)

        if swap:
            dist_swap = _np_distance(embed_anchor, embed_pos, p, eps)
            dist_neg = np.minimum(dist_neg, dist_swap)
        output = np.maximum(self.margin + dist_pos - dist_neg, 0)
        
        if reduction == "mean":
            return torch.mean(output)
        elif reduction == "sum":
            return torch.sum(output)
        else:
            return output

    def forward(self, embed_anchor, embed_pos, embed_neg, embed_nn, dis):
        loss_tml = self.Quadrupletloss(embed_anchor, embed_pos, embed_neg, embed_nn)
        loss_dis = torch.mean(dis)
        loss = self.alpha1 * loss_tml + self.alpha2 * loss_dis
        return loss

class MyLoss_one(nn.Module):
    def __init__(self, alpha1, alpha2, margin):
        super(MyLoss_one, self).__init__()
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.margin = margin
        self.criterion_tml = torch.nn.TripletMarginLoss(margin=margin, p=2)
        return
    
    def forward(self, embed_anchor, embed_pos, embed_neg, disOne):
        loss_tml = self.criterion_tml(embed_anchor, embed_pos, embed_neg)
        loss_dis = torch.mean(disOne)
        loss = self.alpha1 * loss_tml + self.alpha2 * loss_dis
        return loss

class MyLoss_two(nn.Module):
    def __init__(self, alpha1, alpha2, margin):
        super(MyLoss_two, self).__init__()
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.margin = margin
        self.criterion_tml = torch.nn.TripletMarginLoss(margin=margin, p=2)
        return

    def forward(self, embed_anchor, embed_neg, embed_nn, disTwo):
        loss_tml = self.criterion_tml(embed_neg, embed_nn, embed_anchor)
        loss_dis = torch.mean(disTwo)
        loss = self.alpha1 * loss_tml + self.alpha2 * loss_dis
        return loss
