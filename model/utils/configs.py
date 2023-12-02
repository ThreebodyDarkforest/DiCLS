from pydantic import BaseModel
from pydantic_yaml import to_yaml_file, parse_yaml_file_as
from typing import List

def load_config(path):
    return parse_yaml_file_as(Config, path)

def save_config(path, cfg):
    to_yaml_file(path, cfg)

class Lang(BaseModel):
    agg_token: bool = True
    model_name: str = 'bert-base-uncased'
    model_type: str = 'bert'
    tokenizer: None = None
    emb_dim: int = 768
    output_dim: int = 128
    freeze_model: bool = True

class Swint(BaseModel):
    in_channels: List = [192, 384, 768, 768]
    out_channels: int = 768
    use_relu: bool = True
    drop_prob: float = 0.1
    drop_size: int = 4
    drop_block: bool = True
    use_spp: bool = True
    use_pan: bool = True

class Vision(BaseModel):
    model_name: str = 'swint-v2'
    text_feat_dim: int = 768
    output_dim: int = 768
    hidden_dim: int = 2048
    pretrained: bool = True
    freeze_model: bool = True
    frozen_stages: int = 0

    swint: Swint = Swint()
     
class EncoderConfig(BaseModel):
    lang: Lang = Lang()
    visual: Vision = Vision()

class FuseVision(BaseModel):
    embed_dim: int = 768
    num_heads: int = 8
    droppath: float = 0.3
    dropout: float = 0.1
    use_van_attn: bool = False

class FuseLang(BaseModel):
    hidden_size: int = 768
    num_attention_heads: int = 8

class FusionConfig(BaseModel):
    visual: FuseVision = FuseVision()
    lang: FuseLang = FuseLang()
    embed_dim: int = 768
    hidden_dim: int = 768
    channels: int = 197
    num_heads: int = 12
    self_attn: bool = True

    num_layer: int = 2
    drop: List[float] = [0.1, 0.2]

class FocalConfig(BaseModel):
    alpha: float = 0.75
    gamma: float = 2
    reduction: str = 'mean'

    weight: float = 50

class ArcfaceConfig(BaseModel):
    margin: int = 0.7
    scale: int = 64

    weight: float = 0.2

class TokenConfig(BaseModel):
    local_temperature: float = 0.1
    bi_direction: bool = True
    img_encoder: str = 'vit'

    weight: float = 0.05

class ContrasiveConfig(BaseModel):
    softmax_temperature: float = 0.07

    weight: float = 0.01

class PrototypeConfig(BaseModel):
    epsilon = 0.05
    sinkhorn_iterations = 3
    proto_temperature = 0.2

    weight: float = 0.1

class AlignConfig(BaseModel):
    p: float = 1
    alpha = 0.99
    gamma = 1
    positive_temperature = 10

    weight: float = 1

class Loss(BaseModel):
    focal_loss: FocalConfig = FocalConfig()
    arcface_loss: ArcfaceConfig = ArcfaceConfig()
    contrasive_loss: ContrasiveConfig = ContrasiveConfig()
    proto_loss: PrototypeConfig = PrototypeConfig()
    token_loss: TokenConfig = TokenConfig()
    align_loss: AlignConfig = AlignConfig()

class Test(BaseModel):
    accuracy_on: bool = True
    precision_on: bool = True
    recall_on: bool = True
    pr_curve_on: bool = True
    f1_score_on: bool = True
    avg_precision_on: bool = True
    roc_curve_on: bool = True

class Config(BaseModel):
    encoder: EncoderConfig = EncoderConfig()
    fuse: FusionConfig = FusionConfig()
    loss: Loss = Loss()
    eval: Test = Test()

    feature_depth: int = 1
    tokenizer_max_length: int = 197

    local_hidden_dim: int = 768
    local_output_dim: int = 512

    glob_hidden_dim: int = 768
    glob_output_dim: int = 512

    align_dim: int = 256

    num_class: int = 6
    prototypes: int = 128

    use_arcface_loss: bool = True
    use_token_loss: bool = True
    use_proto_loss: bool = True
    use_contrasive_loss: bool = False
    use_align_loss: bool = True

    one_hot: bool = True
    use_ori_classnames: bool = True
    fuse_features: bool = False

    pred_threshold: float = 0.5

    def merge_from_config(self, cfg):
        self.__dict__.update(cfg.__dict__)

    def merge_from_file(self, path):
        cfg = load_config(path)
        self.__dict__.update(cfg.__dict__)

if __name__ == '__main__':
    to_yaml_file('./test.yaml', Config())
    x = parse_yaml_file_as(Config, './test.yaml')
    print(x)