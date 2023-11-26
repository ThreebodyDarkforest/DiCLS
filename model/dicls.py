import torch
import torch.nn as nn
from .backbone import ImageEncoder, LanguageEncoder, FusionBlock
from .utils.loss import (
    LossEvaluator,
    FocalLoss,
    ArcfaceLoss,
    TokenLoss,
    ContrasiveLoss,
    PrototypeLoss,
    AlignLoss
)
from .backbone import LocalEmbedding, GlobalEmbedding
from .models import Mlp
import torch.nn.functional as F
from copy import deepcopy
from einops import rearrange

class DiCLS(nn.Module):
    # A pretty naive model name
    def __init__(self, cfg, is_train: bool = True) -> None:
        super().__init__()
        self.cfg = cfg
        venc_cfg = cfg.encoder.visual
        lenc_cfg = cfg.encoder.lang
        self.num_layer = cfg.fuse.num_layer
        self.is_training = is_train

        self.vis_backbone = ImageEncoder(venc_cfg, 
                                         **venc_cfg.dict())
        
        self.lang_backbone = LanguageEncoder(lenc_cfg,
                                             **lenc_cfg.dict())
        self.tokenizer = self.lang_backbone.tokenizer

        self.fuse_blocks = nn.ModuleList([
            FusionBlock(cfg.fuse, embed_dim=cfg.fuse.embed_dim,
                        hidden_dim=cfg.fuse.hidden_dim, drop=cfg.fuse.drop[i],
                        self_attn=cfg.fuse.self_attn)
            for i in range(self.num_layer)
        ])

        self.pool = nn.AdaptiveAvgPool2d((1, cfg.fuse.embed_dim))

        self.local_emb = LocalEmbedding(cfg.fuse.embed_dim, \
                                        cfg.local_hidden_dim, cfg.local_output_dim)
        self.glob_emb = GlobalEmbedding(cfg.fuse.embed_dim, \
                                        cfg.glob_hidden_dim, cfg.glob_output_dim)
        self.proto_proj = Mlp(cfg.glob_output_dim, out_features=cfg.prototypes)
        self.arc_proj = nn.Linear(cfg.fuse.embed_dim, cfg.tokenizer_max_length)

        #self.head = nn.Linear(cfg.fuse.embed_dim, out_features=cfg.num_class)

        self.loss_evaluater = LossEvaluator(cfg)

        # args_dict = self.cfg.loss.focal_loss.dict()
        # args_dict['one_hot'] = self.cfg.one_hot
        # self.loss = FocalLoss(**args_dict)
    
    def train(self, mode=True):
        self.is_training = mode
        for module in self.children():
            module.train(mode)
        return self
    
    def eval(self):
        return self.train(False)
    
    def forward(self, x, targets=None):
        img, tokens = x
        device = img.device

        visual = self.vis_backbone(img)
        lang = self.lang_backbone(**tokens)

        inputs = { "visual": visual, "lang": lang }

        early_output = inputs["visual"]["all_hidden"].clone()
        early_num = len(early_output)

        visual_outs, lang_outs = [], []
        for i, blk in enumerate(self.fuse_blocks):
            early_feature = early_output[early_num - self.num_layer + i]
            
            inputs = blk(inputs)

            feat_size = inputs["visual"]["hidden"].shape
            early_feature = F.interpolate(early_feature, size=feat_size[-1])

            inputs["visual"]["hidden"] = inputs["visual"]["hidden"] + early_feature
            visual_outs.append(inputs["visual"]["hidden"])
            lang_outs.append(inputs["lang"]["hidden"])

        with torch.no_grad():
            vis_attn = inputs["visual"]["last_attn"].detach()
            lang_attn = inputs["lang"]["last_attn"].detach()
            mask = inputs["lang"]["masks"].detach()

        patch_emb, img_emb = inputs["visual"]["local"], inputs["visual"]["global"]
        word_emb, report_emb = inputs["lang"]["word"], inputs["lang"]["report"]

        patch_emb = self.local_emb(patch_emb)
        patch_emb = F.normalize(patch_emb, dim=-1)
        word_emb = self.local_emb(word_emb)
        word_emb = F.normalize(word_emb, dim=1)

        img_emb = self.glob_emb(img_emb)
        img_emb = F.normalize(img_emb, dim=-1)
        report_emb = self.glob_emb(report_emb)
        report_emb = F.normalize(report_emb, dim=-1)

        img_proto_out = self.proto_proj(img_emb)
        report_proto_out = self.proto_proj(report_emb)

        visual_outs = torch.stack(visual_outs).transpose(0, 2)
        lang_outs = torch.stack(lang_outs).transpose(0, 2)

        vis_feat = self.pool(visual_outs[:, :, -self.cfg.feature_depth:]).squeeze(2)
        vis_feat = vis_feat.transpose(0, 1)
        
        lang_feat = self.pool(lang_outs[:, :, -self.cfg.feature_depth:]).squeeze(2)
        lang_feat = lang_feat.transpose(0, 1)

        #fuse_feat = torch.cat((vis_feat[:, 0], lang_feat[:, 0]), dim=-1).squeeze(1)

        cls_feat = vis_feat[:, 0]
        #logits = self.head(cls_feat)
        #logits_probs = logits.softmax(dim=-1)
        #logits_probs = torch.sigmoid(logits)
        logits = (img_emb.unsqueeze(1) @ word_emb.transpose(-1, -2)).squeeze(1) / self.cfg.loss.align_loss.p
        logits[(mask == 0)[:, 1:].detach()] = float('-inf')
        logits_probs = torch.sigmoid(logits)

        with torch.no_grad():
            w = self.arc_proj.weight.data.clone()
            w = F.normalize(w, dim=1, p=2)
            self.arc_proj.weight.copy_(w)
        
        cls_feat = F.normalize(cls_feat, p=2)
        arc_feat = self.arc_proj(cls_feat).clip(-1, 1)

        if not self.is_training:
            with torch.no_grad():
                sim = (img_emb @ report_emb.t()).detach()
                return inputs, sim, logits_probs

        if self.cfg.use_arcface_loss:
            args_dict = self.cfg.loss.arcface_loss.dict()
            args_dict['one_hot'] = self.cfg.one_hot
            
            inputs = (arc_feat, targets)
            self.loss_evaluater.add_loss(ArcfaceLoss(**args_dict).to(device), inputs, args_dict['weight'])

        if self.cfg.use_token_loss:
            args_dict = self.cfg.loss.token_loss.dict()
    
            # THIS WILL LEAD TO A MEMORY LEAK ON GPU, but I dont know why :(

            #inputs = (word_emb, patch_emb, lang_attn, mask, vis_attn)
            #self.loss_evaluater.add_loss(TokenLoss(**args_dict).to(device), inputs, args_dict['weight'])

            bz = lang_attn.size(0)
            masks = (mask == 0)[:, 1:]
            
            atten_sim = torch.bmm(word_emb, patch_emb.permute(0, 2, 1))
            word_num = word_emb.size(1)
            # atten_sim[mask.unsqueeze(1).repeat(1, word_num, 1)] = float("-inf")
            atten_scores = F.softmax(
                atten_sim / args_dict['local_temperature'], dim=-1)  # bz, 196, 111
            word_atten_output = torch.bmm(atten_scores, patch_emb)

            word_atten_output = F.normalize(word_atten_output, dim=-1)

            with torch.no_grad():
                atten_weights = lang_attn.detach()
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
            
            word_sim = torch.bmm(word_emb, word_atten_output.permute(
                0, 2, 1)) / args_dict['local_temperature']
            
            word_num = word_sim.size(1)
            word_sim_1 = rearrange(word_sim, "b n1 n2 -> (b n1) n2")
            labels = torch.arange(word_num).type_as(
                word_emb).long().repeat(bz)

            loss_word_1 = torch.sum(F.cross_entropy(
                word_sim_1, labels, reduction="none") * word_atten_weights.view(-1)) / bz

            word_sim_2 = rearrange(word_sim, "b n1 n2 -> (b n2) n1")
            loss_word_2 = torch.sum(F.cross_entropy(
                word_sim_2, labels, reduction="none") * word_atten_weights.view(-1)) / bz

            loss_word = (loss_word_1 + loss_word_2) / 2.

            if args_dict['bi_direction']:
                atten_sim = torch.bmm(patch_emb, word_emb.permute(0, 2, 1))
                patch_num = patch_emb.size(1)
                atten_sim[masks.detach().unsqueeze(1).repeat(
                    1, patch_num, 1)] = float("-inf")
                atten_scores = F.softmax(
                    atten_sim / args_dict['local_temperature'], dim=-1)  # bz, 196, 111
                patch_atten_output = torch.bmm(atten_scores, word_emb)

                if "vit" not in args_dict['img_encoder']:
                    patch_atten_output = F.normalize(patch_atten_output, dim=-1)
                    patch_num = patch_atten_output.size(1)
                    patch_atten_weights = torch.ones(
                        bz, patch_num) / patch_num

                else:
                    with torch.no_grad():
                        img_attn_map = vis_attn.detach()
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

                patch_sim = torch.bmm(patch_emb, patch_atten_output.permute(
                    0, 2, 1)) / args_dict['local_temperature']
                
                patch_num = patch_sim.size(1)
                patch_sim_1 = rearrange(patch_sim, "b n1 n2 -> (b n1) n2")
                labels = torch.arange(patch_num).type_as(
                    patch_emb).long().repeat(bz)
                # loss_patch_1 = F.cross_entropy(patch_sim_1, labels)
                loss_patch_1 = torch.sum(F.cross_entropy(
                    patch_sim_1, labels, reduction="none") * patch_atten_weights.view(-1)) / bz

                patch_sim_2 = rearrange(patch_sim, "b n1 n2 -> (b n2) n1")
                loss_patch_2 = torch.sum(F.cross_entropy(
                    patch_sim_2, labels, reduction="none") * patch_atten_weights.view(-1)) / bz

                loss_patch = (loss_patch_1 + loss_patch_2) / 2.

                loss_local = loss_patch + loss_word
            else:
                loss_local = loss_word

            self.loss_evaluater.add_loss(TokenLoss(**args_dict).to(device), (loss_local, ), args_dict['weight'])

        if self.cfg.use_contrasive_loss:
            args_dict = self.cfg.loss.contrasive_loss.dict()
            
            inputs = (img_emb, report_emb)
            self.loss_evaluater.add_loss(ContrasiveLoss(**args_dict).to(device), inputs, args_dict['weight'])

        if self.cfg.use_proto_loss:
            args_dict = self.cfg.loss.proto_loss.dict()
            
            inputs = (img_proto_out, report_proto_out)
            self.loss_evaluater.add_loss(PrototypeLoss(**args_dict).to(device), inputs, args_dict['weight'])
        
        if self.cfg.use_align_loss:
            args_dict = self.cfg.loss.align_loss.dict()

            inputs = (word_emb, img_emb, targets[:, 1:], lang_attn, mask)
            self.loss_evaluater.add_loss(AlignLoss(**args_dict).to(device), inputs, args_dict['weight'])


        args_dict = self.cfg.loss.focal_loss.dict()
        args_dict['one_hot'] = self.cfg.one_hot
        
        inputs = (logits_probs, targets[:, 1:])
        self.loss_evaluater.add_loss(FocalLoss(**args_dict).to(device), inputs, args_dict['weight'])
            
        losses = self.loss_evaluater()
        return losses