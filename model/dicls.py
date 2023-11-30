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
from timm.models.layers import trunc_normal_

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
        self.arc_proj = nn.Linear(cfg.fuse.embed_dim * (2 if cfg.fuse_features else 1), cfg.num_class)

        self.img_proj = nn.Linear(cfg.fuse.embed_dim, out_features=cfg.align_dim)
        self.word_proj = nn.Linear(cfg.fuse.embed_dim, out_features=cfg.align_dim)

        self.head = nn.Linear(cfg.fuse.embed_dim * (2 if cfg.fuse_features else 1), out_features=cfg.num_class)

        self.loss_evaluater = LossEvaluator(cfg)

        # args_dict = self.cfg.loss.focal_loss.dict()
        # args_dict['one_hot'] = self.cfg.one_hot
        # self.loss = FocalLoss(**args_dict)
        #self.img_proj.apply(self._init_weights)
        #self.word_proj.apply(self._init_weights)
        trunc_normal_(self.img_proj.weight, std=.02)
        trunc_normal_(self.word_proj.weight, std=.02)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def train(self, mode=True):
        self.is_training = mode
        for module in self.children():
            module.train(mode)
        return self
    
    def eval(self):
        return self.train(False)
    
    def forward(self, x, targets=None, word_targets=None):
        img, tokens = x
        device = img.device

        visual = self.vis_backbone(img)
        lang = self.lang_backbone(**tokens)

        inputs = { "visual": visual, "lang": lang }

        img_emb_e = inputs["visual"]["global"]
        report_emb_e = inputs["lang"]["report"]

        early_output = inputs["visual"]["all_hidden"]
        early_num = len(early_output)

        visual_outs, lang_outs = [], []
        for i, blk in enumerate(self.fuse_blocks):
            early_feature = early_output[early_num - self.num_layer + i].transpose(-1, -2)
            
            inputs = blk(inputs)

            feat_size = inputs["visual"]["hidden"].shape
            early_feature = F.interpolate(early_feature, size=feat_size[1]).transpose(-1, -2)

            inputs["visual"]["hidden"] = inputs["visual"]["hidden"] + early_feature
            visual_outs.append(inputs["visual"]["hidden"])
            lang_outs.append(inputs["lang"]["hidden"])

        with torch.no_grad():
            vis_attn = inputs["visual"]["last_attn"].detach()
            lang_attn = inputs["lang"]["last_attn"].detach()
            mask = inputs["lang"]["masks"].detach()

        patch_emb = inputs["visual"]["local"]
        word_emb = inputs["lang"]["word"]
        img_emb = inputs["visual"]["global"]
        report_emb = inputs["lang"]["report"]

        word_feat = self.word_proj(word_emb)
        img_feat = self.img_proj(img_emb)

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

        fuse_feat = torch.cat((vis_feat[:, 0], lang_feat[:, 0]), dim=-1).squeeze(1)

        cls_feat = vis_feat[:, 0] #fuse_feat # 
        logits = self.head(cls_feat)
        #logits_probs = logits.softmax(dim=-1)
        logits_probs = torch.sigmoid(logits)

        #logits = (img_emb.unsqueeze(1) @ word_emb.transpose(-1, -2)).squeeze(1) / self.cfg.loss.align_loss.p
        #logits[(mask == 0)[:, 1:].detach()] = float('-inf')
        #logits_probs = torch.sigmoid(logits)

        with torch.no_grad():
            w = self.arc_proj.weight.data.clone()
            w = F.normalize(w, dim=1, p=2)
            self.arc_proj.weight.copy_(w)
        
        cls_feat = F.normalize(cls_feat, p=2)
        arc_feat = self.arc_proj(cls_feat).clip(-1, 1)

        if not self.is_training:
            with torch.no_grad():
                sim = torch.sigmoid((img_emb @ report_emb.t()).detach())
                return inputs, sim, logits_probs

        if self.cfg.use_arcface_loss:
            args_dict = self.cfg.loss.arcface_loss.dict()
            
            inputs = (arc_feat, targets)
            self.loss_evaluater.add_loss('ArcfaceLoss', inputs, args_dict['weight'])

        if self.cfg.use_token_loss:
            args_dict = self.cfg.loss.token_loss.dict()

            inputs = (word_emb, patch_emb, lang_attn, mask, vis_attn)
            self.loss_evaluater.add_loss('TokenLoss', inputs, args_dict['weight'])

        if self.cfg.use_contrasive_loss:
            args_dict = self.cfg.loss.contrasive_loss.dict()
            
            inputs = (img_emb_e, report_emb_e)
            self.loss_evaluater.add_loss('ContrasiveLoss', inputs, args_dict['weight'])

        if self.cfg.use_proto_loss:
            args_dict = self.cfg.loss.proto_loss.dict()
            
            inputs = (img_proto_out, report_proto_out)
            self.loss_evaluater.add_loss('PrototypeLoss', inputs, args_dict['weight'])
        
        if self.cfg.use_align_loss:
            args_dict = self.cfg.loss.align_loss.dict()

            inputs = (word_feat, img_feat, word_targets[:, 1:], lang_attn, mask)
            self.loss_evaluater.add_loss('AlignLoss', inputs, args_dict['weight'])


        args_dict = self.cfg.loss.focal_loss.dict()
        
        inputs = (logits_probs, targets)
        self.loss_evaluater.add_loss('FocalLoss', inputs, args_dict['weight'])
            
        losses = self.loss_evaluater()
        return losses