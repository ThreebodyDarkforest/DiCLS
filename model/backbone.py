import torch
import torch.nn as nn
import torch.functional as F
import transformers
#from torchvision.models import VisionTransformer, SwinTransformer
from transformers.models.gpt2 import GPT2Model, GPT2Tokenizer, GPT2Config
from transformers.models.bert import BertConfig, BertModel, BertTokenizer, BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertAttention, BertIntermediate, BertOutput
from transformers.modeling_utils import apply_chunking_to_forward
from transformers import AutoTokenizer, AutoConfig
from einops import rearrange
import os
import math
from . import cnn
from .vits import VisionTransformer, Block
from .van import Block as VanBlock
from .models import Mlp
from timm.models.layers import trunc_normal_

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def _vision_backbone(vit, image_size, use_grad_checkpointing=False, ckpt_layer=0, drop_path_rate=0):
    assert vit in ['base', 'large'], "vit parameter must be base or large"
    if vit == 'base':
        vision_width = 768
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=12,
                                           num_heads=12, use_grad_checkpointing=use_grad_checkpointing, ckpt_layer=ckpt_layer,
                                           drop_path_rate=0 or drop_path_rate
                                           )
    elif vit == 'large':
        vision_width = 1024
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=24,
                                           num_heads=16, use_grad_checkpointing=use_grad_checkpointing, ckpt_layer=ckpt_layer,
                                           drop_path_rate=0.1 or drop_path_rate
                                           )
    return visual_encoder, vision_width

def _language_backbone(cfg, model_name, model_type):
    cfg_ = GPT2Config.from_pretrained(model_name) if 'gpt' in model_type else BertConfig.from_pretrained(model_name)
    assert model_type in ('gpt', 'bert')
    if 'gpt' in model_type:
        model = GPT2Model.from_pretrained(model_name, config=cfg_)
    elif 'bert' in model_type:
        model = BertModel.from_pretrained(model_name, config=cfg_, add_pooling_layer=False)
    return model

class GlobalEmbedding(nn.Module):
    def __init__(self, input_dim=768,
                 hidden_dim=2048,
                 output_dim=512) -> None:
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim, affine=False)  # output layer
        )
    
    def forward(self, x):
        return self.head(x)

class LocalEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, output_dim,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(output_dim, affine=False)  # output layer
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.head(x)

        return x.permute(0, 2, 1)

class ImageEncoder(nn.Module):
    def __init__(self, cfg,
                 model_name: str = "resnet_50",
                 text_feat_dim: int = 768,
                 output_dim: int = 768,
                 hidden_dim: int = 2048,
                 pretrained: bool = True,
                 freeze_model: bool = True,
                 **kwargs
                 ):
        super(ImageEncoder, self).__init__()

        self.model_name = model_name
        self.output_dim = output_dim
        self.text_feat_dim = text_feat_dim
        self.freeze_model = freeze_model

        if "vit" in model_name:
            vit_grad_ckpt = False
            vit_ckpt_layer = 0
            image_size = 224

            vit_name = model_name[4:]
            self.model, vision_width = _vision_backbone(
                vit_name, image_size, vit_grad_ckpt, vit_ckpt_layer, 0)

            self.feature_dim = vision_width

            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True)
            state_dict = checkpoint["model"]
            self.model.load_state_dict(state_dict, strict=False)

            # TODO: add some extra network like PAN

        else:
            model_function = getattr(
                cnn, model_name)
            self.model, self.feature_dim, self.interm_feature_dim = model_function(
                pretrained=pretrained
            )

            # Average pooling
            self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        
        if self.freeze_model is True:
            print("Freezing Vision backbone...")
            for param in self.model.parameters():
                param.requires_grad = False

    def resnet_forward(self, x, get_local=True):
        x = nn.Upsample(size=(299, 299), mode="bilinear",
                        align_corners=True)(x)
        x = self.model.conv1(x)  # (batch_size, 64, 150, 150)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)\
        
        all_hidden = []

        x = self.model.layer1(x)  # (batch_size, 64, 75, 75)
        all_hidden.append(x)
        x = self.model.layer2(x)  # (batch_size, 128, 38, 38)
        all_hidden.append(x)
        x = self.model.layer3(x)  # (batch_size, 256, 19, 19)
        all_hidden.append(x)
        x = self.model.layer4(x)  # (batch_size, 512, 10, 10)
        all_hidden.append(x)

        all_hidden = [hidden.clone().detach() for hidden in all_hidden]

        local_features = x

        x = self.pool(x)
        x = x.view(x.size(0), -1)

        local_features = rearrange(local_features, "b c w h -> b (w h) c")

        #return x, local_features.contiguous()
        outputs = {
            "global": x,
            "local": local_features.contiguous(),
            "hidden": torch.cat((x.unsqueeze(0), local_features), dim=1),
            "all_hidden": all_hidden
        }
        outputs.update({ "aggregate": outputs["global"].clone(), "embedded": outputs["local"].clone() })
        return outputs

    def vit_forward(self, x):
        return self.model(x, register_blk=11)

    def forward(self, x, get_local=False):
        if "resnet" in self.model_name:
            return self.resnet_forward(x, get_local=get_local)
        elif "vit" in self.model_name:
            img_feat, hidden_states = self.vit_forward(x)
            #return img_feat[:, 0].contiguous(), img_feat[:, 1:].contiguous()
            return {
                "global": img_feat[:, 0].contiguous(),
                "local": img_feat[:, 1:].contiguous(),
                "hidden": img_feat,
                "all_hidden": hidden_states,
                "aggregate": img_feat[:, 0].contiguous(),
                "embedded": img_feat[:, 1:].contiguous(),
                "last_attn": self.model.blocks[11].attn.attention_map,
            }

class LanguageEncoder(nn.Module):
    def __init__(self, cfg, agg_tokens=True,
                 model_name = None,
                 model_type = 'gpt2',
                 tokenizer = None,
                 emb_dim: int = 768,
                 output_dim: int = 128,
                 freeze_model: bool = True,
                 **kwargs) -> None:
        super().__init__()
        self.cfg = cfg
        self.model_type = model_type
        self.embedding_dim = emb_dim
        self.output_dim = output_dim
        self.freeze_model = freeze_model
        self.agg_tokens = agg_tokens

        self.model = _language_backbone(cfg, model_name, model_type)
        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = '[PAD]'

        self.idx2word = {v : k for k, v in self.tokenizer.get_vocab().items()}
        if self.freeze_model is True:
            print("Freezing Language backbone...")
            for param in self.model.parameters():
                param.requires_grad = False

    def aggregate_tokens(self, embeddings, caption_ids, last_layer_attn):
        _, num_layers, num_words, dim = embeddings.shape
        embeddings = embeddings.permute(0, 2, 1, 3)
        agg_embs_batch = []
        sentences = []
        last_attns = []

        # loop over batch
        for embs, caption_id, last_attn in zip(embeddings, caption_ids, last_layer_attn):
            agg_embs = []
            token_bank = []
            words = []
            word_bank = []
            attns = []
            attn_bank = []

            # loop over sentence
            for word_emb, word_id, attn in zip(embs, caption_id, last_attn):
                word = self.idx2word[word_id.item()]
                if word == "[SEP]":
                    new_emb = torch.stack(token_bank)
                    new_emb = new_emb.sum(axis=0)
                    agg_embs.append(new_emb)
                    words.append("".join(word_bank))
                    attns.append(sum(attn_bank))
                    agg_embs.append(word_emb)
                    words.append(word)
                    attns.append(attn)
                    break
                # This is because some words are divided into two words.
                if not word.startswith("##"):
                    if len(word_bank) == 0:
                        token_bank.append(word_emb)
                        word_bank.append(word)
                        attn_bank.append(attn)
                    else:
                        new_emb = torch.stack(token_bank)
                        new_emb = new_emb.sum(axis=0)
                        agg_embs.append(new_emb)
                        words.append("".join(word_bank))
                        attns.append(sum(attn_bank))

                        token_bank = [word_emb]
                        word_bank = [word]
                        attn_bank = [attn]
                else:
                    token_bank.append(word_emb)
                    word_bank.append(word[2:])
                    attn_bank.append(attn)
            agg_embs = torch.stack(agg_embs)
            padding_size = num_words - len(agg_embs)
            paddings = torch.zeros(padding_size, num_layers, dim)
            paddings = paddings.type_as(agg_embs)
            words = words + ["[PAD]"] * padding_size
            last_attns.append(
                torch.cat([torch.tensor(attns), torch.zeros(padding_size)], dim=0))
            agg_embs_batch.append(torch.cat([agg_embs, paddings]))
            sentences.append(words)

        agg_embs_batch = torch.stack(agg_embs_batch)
        agg_embs_batch = agg_embs_batch.permute(0, 2, 1, 3)
        last_atten_pt = torch.stack(last_attns)
        last_atten_pt = last_atten_pt.type_as(agg_embs_batch)

        return agg_embs_batch, sentences, last_atten_pt
    
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        if self.model_type == 'gpt':
            outputs = self.model(input_ids, attention_mask=attention_mask, return_dict=True,
                                 output_attentions=True, output_hidden_states=True)
        else:
            outputs = self.model(input_ids, attention_mask, token_type_ids, return_dict=True,
                                 output_attentions=True, output_hidden_states=True)
        last_layer_attn = outputs.attentions[-1][:, :, 0, 1:].mean(dim=1)
        all_feat = outputs.last_hidden_state.unsqueeze(1)
        num_layers = all_feat.shape[1]
        all_feat, sents, last_atten_pt = self.aggregate_tokens(
                all_feat, input_ids, last_layer_attn)
        last_atten_pt = last_atten_pt[:, 1:].contiguous()
        
        all_feat = all_feat[:, 0]
        report_feat = all_feat[:, 0].contiguous()
        word_feat = all_feat[:, 1:].contiguous()

        encoded_layers = outputs.hidden_states[1:]

        features = None
        features = torch.stack(encoded_layers[-num_layers:], 1).mean(1)
        # language embedding has shape [len(phrase), seq_len, language_dim]
        features = features / num_layers

        embedded = features * attention_mask.unsqueeze(-1).float()
        aggregate = embedded.sum(1) / (attention_mask.sum(-1).unsqueeze(-1).float())
        
        #return report_feat, word_feat, last_atten_pt
        return {
            "aggregate": aggregate,
            "embedded": embedded,
            "masks": attention_mask,
            "hidden": encoded_layers[-1],
            "all_hidden": encoded_layers,
            "report": report_feat,
            "word": word_feat,
            "last_attn": last_atten_pt
        }

class BertEncoderLayer(BertPreTrainedModel):
    def __init__(self, config, position_embedding_type=None):
        super().__init__(config)
        self.config = config

        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1

        self.attention = BertAttention(config, position_embedding_type)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
        self.gate = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.act_fn = nn.GELU()

        trunc_normal_(self.gate, std=.02)

    def forward(self, inputs):
        language_dict_features = inputs["lang"]
        hidden_states = language_dict_features["hidden"]
        attention_mask = language_dict_features["masks"]

        device = hidden_states.device
        input_shape = hidden_states.size()[:-1]
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device)

        self_attention_outputs = self.attention(
            hidden_states,
            extended_attention_mask,
            None,
            output_attentions=True,
            past_key_value=None,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs
        hidden_states = outputs[0]

        hidden_states = self.act_fn(hidden_states + torch.tanh(self.gate))
        last_attn_pt = outputs[1][:, :, 0, 1:].mean(dim=1)

        language_dict_features = {
            "hidden": hidden_states,
            "masks": attention_mask,
            "report": hidden_states[:, 0].contiguous(),
            "word": hidden_states[:, 1:].contiguous(),
            "last_attn": last_attn_pt
        }

        features_dict = {
            "visual": inputs["visual"],
            "lang": language_dict_features
        }

        return features_dict

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class VisionEncoderLayer(nn.Module):
    def __init__(self, cfg, embed_dim, num_heads=8, droppath=0.3,
                 dropout=0.1, use_van_attn=False) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.droppath = droppath
        self.dropout = dropout
        self.use_van = use_van_attn

        self.encoder = VanBlock(self.embed_dim, drop_path = self.droppath) if use_van_attn else \
                       Block(self.embed_dim, self.num_heads, drop_path = self.droppath)
        self.gate = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.dropout = nn.Dropout(self.dropout)
        
        trunc_normal_(self.gate, std=.02)

    def forward(self, inputs):
        features = inputs['visual']['hidden']
        
        if self.use_van:
            patch_num = int(features.size(1) ** 0.5)
            features = features[:, 1:, :].transpose(-1, -2).view(-1, self.embed_dim, patch_num, patch_num)

        features = self.encoder(features, register_hook=True)
        last_attn_pt = self.encoder.attn.attention_map
        if self.use_van:
            last_attn_pt = last_attn_pt.view(-1, self.embed_dim, patch_num * patch_num)
            last_attn_pt = last_attn_pt.mean(dim=1)
        else:
            last_attn_pt = last_attn_pt[:, :, 0, 1:].mean(dim=1)

        if self.use_van:
            features = features.view(-1, features.size(1), patch_num * patch_num)
            features = torch.cat((features.mean(dim=-1, keepdim=True), features), dim=-1)
            features = features.transpose(-1, -2)

        features = features + torch.tanh(self.gate)
        features = self.dropout(features)

        features = { 
            "global": features[:, 0].contiguous(),
            "local": features[:, 1:].contiguous(),
            "hidden": features,
            "last_attn": last_attn_pt
        }

        return {
            "visual": features,
            "lang": inputs['lang']
        }

class CrossAttention(nn.Module):
    def __init__(self, cfg, in_dim, embed_dim, num_heads=8, drop=0.3, self_attn=True) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.self_attn = self_attn
        self.q_len = int(embed_dim // num_heads)

        self.kv_proj = nn.Linear(in_dim, 2 * embed_dim)
        self.q_proj = nn.Linear(in_dim, embed_dim)

        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _head_transpose(self, x: torch.Tensor):
        return x.view(-1, x.size(1), self.num_heads, x.size(2) // self.num_heads).permute((0, 2, 1, 3))

    def forward(self, x: torch.Tensor, query: torch.Tensor):
        B, N, C = x.shape
        kv = self.kv_proj(x)    # bz, c, 2 * embed_dim 
        kv = kv.permute((2, 0, 1)).view(2, kv.size(2) // 2, -1, kv.size(1))
        kv = kv.permute((0, 2, 3, 1))
        q = self.q_proj(query)

        k, v = kv[0], kv[1]
        if self.self_attn:
            k, v = torch.cat((k, q), dim=1), torch.cat((v, v), dim=-2)
        
        q, k, v = [self._head_transpose(x) for x in [q, k, v]]

        attn_scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.q_len)

        attn_probs = attn_scores.softmax(dim=-1)

        attn_weights = (attn_probs @ v).transpose(1, 2).contiguous()
        attn_weights = attn_weights.view(-1, attn_weights.size(1), self.embed_dim)
        outs = self.drop(attn_weights)

        return outs
    
class BiAttnBlock(nn.Module):
    def __init__(self, cfg, embed_dim, hidden_dim, num_heads, drop, self_attn=False) -> None:
        super().__init__()
        
        self.i2t_attn = CrossAttention(cfg, embed_dim, hidden_dim, num_heads, drop, self_attn)
        self.t2i_attn = CrossAttention(cfg, embed_dim, hidden_dim, num_heads, drop, self_attn)

        self.i2t_head = Mlp(hidden_dim, out_features=embed_dim)
        self.t2i_head = Mlp(hidden_dim, out_features=embed_dim)

    def forward(self, inputs):
        visual = inputs["visual"]["hidden"]
        lang = inputs["lang"]["hidden"]

        inputs["visual"]["hidden"] = self.i2t_head(self.t2i_attn(visual, lang))
        inputs["lang"]["hidden"] = self.t2i_head(self.i2t_attn(lang, visual))

        return inputs

class FusionBlock(nn.Module):
    def __init__(self, cfg, embed_dim=768, hidden_dim=1024, drop=0.3, self_attn=False) -> None:
        super().__init__()

        self.lang_attn_gate = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.visual_attn_gate = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cross_encoder = nn.ModuleList([
            BiAttnBlock(cfg, embed_dim, hidden_dim, cfg.num_heads, drop, self_attn),
            VisionEncoderLayer(cfg.visual, **cfg.visual.dict()),
            BertEncoderLayer(BertConfig(**cfg.lang.dict()), position_embedding_type='absolute'),
        ])
        self.norm_layer1 = nn.BatchNorm1d(embed_dim)
        self.norm_layer2 = nn.BatchNorm1d(embed_dim)
        self.dropout = nn.Dropout(drop)

    def forward(self, inputs):
        outputs = self.cross_encoder[0](inputs)
        outputs = self.cross_encoder[1](outputs)
        outputs = self.cross_encoder[2](outputs)

        outputs["visual"]["hidden"] = self.dropout(self.norm_layer1(
                                      outputs["visual"]["hidden"].transpose(-1, -2)).transpose(-1, -2))
        outputs["lang"]["hidden"] = self.dropout(self.norm_layer2(
                                    outputs["lang"]["hidden"].transpose(-1, -2)).transpose(-1, -2))

        return outputs


if __name__ == '__main__':
    test = LanguageEncoder(None, model_name='bert-base-uncased', model_type='bert')
    #tokenizer = BertTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    #test2 = BertEncoderLayer(BertConfig(), 'absolute')
    #test3 = BertEncoderLayer(BertConfig(), 'absolute')
    x = "hello, ass!"
    x = test.tokenizer(
        x,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=197,
    )
    x = test(**x)
    # print(x['hidden'].shape)
    # x = {"visual": None, "lang": x}
    # print(x['lang']['hidden'].shape)
    # x = test2(x)
    # print(x['lang']['hidden'].shape)
    # x = test3(x)
    # print(x['lang']['hidden'].shape)
    #print(x)
    #print(len(x[-1][1:]))

    test2 = ImageEncoder(None, 'vit-base')
    #test2 = VisionEncoderLayer(None, 768)
    y = torch.randn((1, 3, 224, 224))
    y = test2(y)
    
    # print(x['global'].unsqueeze(0).shape, x['local'].shape, x['hidden'].shape)
    # x = {"visual": x, "lang": None}
    # x = test2(x)
    # print(x['visual']['local'].shape, x['visual']['global'].shape)
    #print(x)

    inputs = {"visual": y, "lang": x}

    config = BertConfig(hidden_size=1024, num_attention_heads=8)
    test3 = FusionBlock(config)
    outs = test3(inputs)
    print(outs)
    # x = torch.randn((1, 256, 768))
    # attn = CrossAttention(None, 768, 1024)
    # x = attn(x, x)
    # print(x.shape)