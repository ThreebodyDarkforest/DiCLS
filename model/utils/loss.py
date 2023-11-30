from typing import Any, Optional
from torch.nn import functional as F
import torch
import torch.nn as nn
from typing import List, Tuple, Any, Dict
from einops import rearrange
from torch.nn.modules.module import Module
import math

__all__ = ['FocalLoss', 'ArcfaceLoss', 'TokenLoss', 'ContrasiveLoss', 'PrototypeLoss', 'AlignLoss']

class LossEvaluator(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        #self.args_list = []
        self.cfg = cfg
        self.loss_dict = {}
        self.losses_fn = nn.ModuleDict()
        # TODO: init loss module

        if self.cfg.use_arcface_loss:
            args_dict = self.cfg.loss.arcface_loss.dict()
            args_dict['one_hot'] = self.cfg.one_hot
            
            self.losses_fn.update({'ArcfaceLoss': ArcfaceLoss(**args_dict)})

        if self.cfg.use_token_loss:
            args_dict = self.cfg.loss.token_loss.dict()
    
            # THIS WILL LEAD TO A MEMORY LEAK ON GPU, but I dont know why :(

            self.losses_fn.update({'TokenLoss': TokenLoss(**args_dict)})

        if self.cfg.use_contrasive_loss:
            args_dict = self.cfg.loss.contrasive_loss.dict()

            self.losses_fn.update({'ContrasiveLoss': ContrasiveLoss(**args_dict)})

        if self.cfg.use_proto_loss:
            args_dict = self.cfg.loss.proto_loss.dict()
            
            self.losses_fn.update({'PrototypeLoss': PrototypeLoss(**args_dict)})
        
        if self.cfg.use_align_loss:
            args_dict = self.cfg.loss.align_loss.dict()

            self.losses_fn.update({'AlignLoss': AlignLoss(**args_dict)})
        
        args_dict = self.cfg.loss.focal_loss.dict()
        args_dict['one_hot'] = self.cfg.one_hot
        
        self.losses_fn.update({'FocalLoss': FocalLoss(**args_dict)})

    def add_loss(self, name, inputs, weight):
        self.loss_dict.update({name : (inputs, weight)})

    def forward(self):
        losses = {}
        for name, (inputs, weight) in self.loss_dict.items():
            loss = self.losses_fn[name](*inputs) * weight
            losses.update({name : loss})
        
        loss = [l for l in losses.values()]
        loss = torch.sum(torch.stack(loss))
        losses.update({"total" : loss})

        return losses

class AbstractLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x

class FocalLoss(AbstractLoss):
    def __init__(self, alpha=0.75, gamma=2, one_hot=False, 
                 reduction='mean', **kwargs):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.one_hot = one_hot

    def forward(self, inputs, targets):
        if self.one_hot:
            ce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class ArcfaceLoss(AbstractLoss):
    def __init__(self, margin=0.7, scale=64, one_hot=False, **kwargs) -> None:
        super(ArcfaceLoss, self).__init__()
        self.margin = margin
        self.scale = scale
        self.one_hot = one_hot

        #self.proj = nn.Linear(in_features, out_features, bias=False)

    def forward(self, cos_theta, targets):
        one_hot = targets if self.one_hot else \
                  F.one_hot(targets)

        arc_cos = torch.acos(cos_theta)

        m = one_hot * self.margin
        arc_cos = arc_cos + m

        cos_theta = torch.cos(arc_cos)
        logits = cos_theta * self.scale

        loss = F.binary_cross_entropy_with_logits(logits, one_hot)
        
        return loss

class TokenLoss(AbstractLoss):
    def __init__(self, local_temperature=0.1, bi_direction=True, 
                 img_encoder='vit-base', **kwargs) -> None:
        super(TokenLoss, self).__init__()

        self.img_encoder = img_encoder
        self.bi_direction = bi_direction
        self.local_temperature = local_temperature

    def forward(self, word_emb_q, patch_emb_q, word_attn_q, mask, patch_attn_q=None) -> Any:
        bz = word_attn_q.size(0)
        mask = (mask == 0)[:, 1:]
        
        atten_sim = torch.bmm(word_emb_q, patch_emb_q.permute(0, 2, 1))
        word_num = word_emb_q.size(1)
        # atten_sim[mask.unsqueeze(1).repeat(1, word_num, 1)] = float("-inf")
        atten_scores = F.softmax(
            atten_sim / self.local_temperature, dim=-1)  # bz, 196, 111
        word_atten_output = torch.bmm(atten_scores, patch_emb_q)

        word_atten_output = F.normalize(word_atten_output, dim=-1)

        with torch.no_grad():
            atten_weights = word_attn_q.detach()
            word_atten_weights = []
            for i in range(bz):
                atten_weight = atten_weights[i]
                nonzero = atten_weight.nonzero().squeeze()
                low = torch.quantile(atten_weight[nonzero], 0.1)
                high = torch.quantile(atten_weight[nonzero], 0.9)
                atten_weight[nonzero] = atten_weight[nonzero].clip(low, high)
                word_atten_weights.append(atten_weight.clone())
            word_atten_weights = torch.stack(word_atten_weights)
        
        word_atten_weights /= word_atten_weights.sum(dim=1, keepdims=True)
        
        word_sim = torch.bmm(word_emb_q, word_atten_output.permute(
            0, 2, 1)) / self.local_temperature
        
        word_num = word_sim.size(1)
        word_sim_1 = rearrange(word_sim, "b n1 n2 -> (b n1) n2")
        targets = torch.arange(word_num).type_as(
            word_emb_q).long().repeat(bz)

        loss_word_1 = torch.sum(F.cross_entropy(
            word_sim_1, targets, reduction="none") * word_atten_weights.view(-1)) / bz

        word_sim_2 = rearrange(word_sim, "b n1 n2 -> (b n2) n1")
        loss_word_2 = torch.sum(F.cross_entropy(
            word_sim_2, targets, reduction="none") * word_atten_weights.view(-1)) / bz

        loss_word = (loss_word_1 + loss_word_2) / 2.

        if self.bi_direction:
            atten_sim = torch.bmm(patch_emb_q, word_emb_q.permute(0, 2, 1))
            patch_num = patch_emb_q.size(1)
            atten_sim[mask.detach().unsqueeze(1).repeat(
                1, patch_num, 1)] = float("-inf")
            atten_scores = F.softmax(
                atten_sim / self.local_temperature, dim=-1)  # bz, 196, 111
            patch_atten_output = torch.bmm(atten_scores, word_emb_q)

            if "vit" not in self.img_encoder:
                patch_atten_output = F.normalize(patch_atten_output, dim=-1)
                patch_num = patch_atten_output.size(1)
                patch_atten_weights = torch.ones(
                    bz, patch_num) / patch_num

            else:
                with torch.no_grad():
                    img_attn_map = patch_attn_q.detach()
                    atten_weights = img_attn_map#[:, :, 0, 1:].mean(dim=1)
                    patch_atten_weights = []
                    for i in range(bz):
                        atten_weight = atten_weights[i]
                        atten_weight = atten_weight.clip(torch.quantile(
                            atten_weight, 0.1), torch.quantile(atten_weight, 0.9))
                        patch_atten_weights.append(atten_weight.clone())
                    patch_atten_weights = torch.stack(patch_atten_weights)

                patch_atten_weights /= patch_atten_weights.sum(
                    dim=1, keepdims=True)

            patch_sim = torch.bmm(patch_emb_q, patch_atten_output.permute(
                0, 2, 1)) / self.local_temperature
            
            patch_num = patch_sim.size(1)
            patch_sim_1 = rearrange(patch_sim, "b n1 n2 -> (b n1) n2")
            targets = torch.arange(patch_num).type_as(
                patch_emb_q).long().repeat(bz)
            # loss_patch_1 = F.cross_entropy(patch_sim_1, targets)
            loss_patch_1 = torch.sum(F.cross_entropy(
                patch_sim_1, targets, reduction="none") * patch_atten_weights.view(-1)) / bz

            patch_sim_2 = rearrange(patch_sim, "b n1 n2 -> (b n2) n1")
            loss_patch_2 = torch.sum(F.cross_entropy(
                patch_sim_2, targets, reduction="none") * patch_atten_weights.view(-1)) / bz

            loss_patch = (loss_patch_1 + loss_patch_2) / 2.

            loss_local = loss_patch + loss_word
        else:
            loss_local = loss_word

        return loss_local

    #def forward(self, x):
    #    return x

class ContrasiveLoss(AbstractLoss):
    def __init__(self, softmax_temperature, **kwargs) -> None:
        super().__init__()
        self.p = softmax_temperature
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        sim1 = x1 @ x2.t() / self.p
        sim2 = sim1.transpose(-1, -2) / self.p

        targets = torch.arange(x1.size(0)).type_as(x1).long()
        loss1 = F.cross_entropy(sim1, targets)
        loss2 = F.cross_entropy(sim2, targets)

        return (loss1 + loss2) / 2.
    
class AlignLoss(AbstractLoss):
    def __init__(self, p, alpha, gamma, positive_temperature, **kwargs) -> None:
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.gamma = gamma
        self.positive_temperature = positive_temperature
    
    def forward(self, word_emb, img_emb, targets, word_attn, mask):
        bz = word_attn.size(0)
        length = torch.sum((mask[:, 1:] == 1)) // mask.size(0)

        mask = (mask == 0)[:, 1:]
        # (bz, 1, hidden_dim) @ (bz, word_num, hidden_dim)
        sim = (img_emb.unsqueeze(1) @ word_emb.transpose(-1, -2)) / self.p
        sim = sim.squeeze(1)
        
        sim = sim[:, :length]
        #sim[mask.detach()] = 0
        sim = torch.sigmoid(sim)

        # with torch.no_grad():
        #     atten_weights = word_attn.detach()
        #     word_atten_weights = []
        #     for i in range(bz):
        #         atten_weight = atten_weights[i]
        #         nonzero = atten_weight.nonzero().squeeze()
        #         low = torch.quantile(atten_weight[nonzero], 0.1)
        #         high = torch.quantile(atten_weight[nonzero], 0.9)
        #         atten_weight[nonzero] = atten_weight[nonzero].clip(low, high)
        #         word_atten_weights.append(atten_weight.clone())
        #     word_atten_weights = torch.stack(word_atten_weights)
        
        # word_atten_weights /= word_atten_weights.sum(dim=1, keepdims=True)

        #print(word_atten_weights[0])
        #print(sim[0])
        #print(length)

        loss = F.binary_cross_entropy_with_logits(
            sim, targets[:, :length], reduction="none")
        
        #print(sim[0], targets[0, :length], loss[0])
        #loss = F.binary_cross_entropy_with_logits(sim, targets)
        
        loss[targets[:, :length] == 0] /= self.positive_temperature
        #pt = torch.exp(-loss)
        #loss = self.alpha * (1 - pt) ** self.gamma * loss
        
        loss = torch.sum(loss) / bz
        
        return loss

class PrototypeLoss(AbstractLoss):
    def __init__(self, epsilon=0.05, sinkhorn_iterations=3, 
                 proto_temperature=0.2, **kwargs) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.sinkhorn_iterations = sinkhorn_iterations
        self.proto_temperature = proto_temperature

        self.gpus = torch.cuda.device_count()
        self.get_assignments = self.sinkhorn

    def forward(self, img_proto_out, report_proto_out):
        with torch.no_grad():
            img_code = torch.exp(
                img_proto_out / self.epsilon).t()
            img_code = self.get_assignments(
                img_code, self.sinkhorn_iterations)         # bz, 500
            report_code = torch.exp(
                report_proto_out / self.epsilon).t()
            report_code = self.get_assignments(
                report_code, self.sinkhorn_iterations)       # bz, 500

        img_proto_prob = F.softmax(
            img_proto_out / self.proto_temperature, dim=1)
        report_proto_prob = F.softmax(
            report_proto_out / self.proto_temperature, dim=1)

        loss_i2t_proto = - \
            torch.mean(torch.sum(img_code * 
                       torch.log(report_proto_prob), dim=1))
        loss_t2i_proto = - \
            torch.mean(torch.sum(report_code *
                       torch.log(img_proto_prob), dim=1))

        loss_proto = (loss_i2t_proto + loss_t2i_proto) / 2.

        return loss_proto

    def sinkhorn(self, Q, nmb_iters):
        ''' 
            :param Q: (num_prototypes, batch size)

        '''
        with torch.no_grad():
            sum_Q = torch.sum(Q)
            Q /= sum_Q

            K, B = Q.shape

            if self.gpus > 0:
                u = torch.zeros(K).cuda()
                r = torch.ones(K).cuda() / K
                c = torch.ones(B).cuda() / B
            else:
                u = torch.zeros(K)
                r = torch.ones(K) / K
                c = torch.ones(B) / B

            for _ in range(nmb_iters):
                u = torch.sum(Q, dim=1)
                Q *= (r / u).unsqueeze(1)
                Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)

            return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()