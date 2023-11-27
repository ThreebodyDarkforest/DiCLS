from model.dicls import DiCLS
from model.utils.configs import Config
from model.backbone import GlobalEmbedding
from dataset.utils import create_dataloader, label2caption, get_caption
from dataset.transform import GeneralTransform
from model.utils.visualizer import vis_grid_attention
import torch
from argparse import ArgumentParser
from PIL import Image
import numpy as np

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--weight', default=None, help='test model path.')
    parser.add_argument('--config', default=None, help='model config file.')
    parser.add_argument('--img', default=None, help='image to inference.')

    args = parser.parse_args()
    cfg = Config()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = DiCLS(cfg).to(device)

    weight_name = args.weight
    checkpoint = torch.load(weight_name)
    model.load_state_dict(checkpoint['model'])

    test_dataloader = create_dataloader('PlantCLS', 'data/plant_dataset', 1, True, dataset_type='test', use_npz=True)
    idx2label = test_dataloader.dataset.idx2label

    with torch.no_grad():
        model.eval()

        img = Image.open(args.img)
        img = GeneralTransform.test_transform(img)
        img = img.unsqueeze(0)

        txt = ["All:" + ",".join([x for x in idx2label.values()])]
        #txt = ["powdery_mildew"]
        #print(txt)
        txt = model.tokenizer(
            txt,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=cfg.tokenizer_max_length,
        )

        res = model((img.to(device), txt.to(device)))
        vis_attn = res[0]['visual']['last_attn'].squeeze()
        vis_attn = vis_attn.view(int(vis_attn.size(0) ** 0.5), int(vis_attn.size(0) ** 0.5))
        print(vis_attn.detach().cpu().numpy())
        vis_attn = vis_attn.detach().cpu().numpy()
        vis_grid_attention(args.img, vis_attn)
        #print(res[-1].detach().cpu().numpy())
        #print(idx2label[np.argmax(res[-1].detach().cpu().numpy())])