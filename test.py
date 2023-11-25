from model.dicls import DiCLS
from model.utils.configs import Config
from model.backbone import GlobalEmbedding
from dataset.utils import create_dataloader, label2caption, get_caption
from model.utils.loss import TokenLoss
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        x1, x2, x3, x4, x5 = torch.randn((32, 196, 256)), torch.randn((32, 196, 256)), torch.randn((32, 196)), torch.randn((32, 197)), torch.randn((32, 196))
        x1 = x1.to(device)
        x2 = x2.to(device)
        x3 = x3.to(device)
        x4 = x4.to(device)
        x5 = x5.to(device)
        x = x.mean()
        return x1 + x, x2 + x, x3 + x, x4 + x, x5 + x

if __name__ == '__main__':
    cfg = Config()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_dataloader = create_dataloader('PlantCLS', './data/plant_dataset', 32, True, dataset_type='train', use_npz=True)

    loss_fn = TokenLoss().to(device)
    model = Net().to(device)

    epoch = 20
    for e in range(epoch):
        for i, data in enumerate(train_dataloader):
            img, target = data
            #x1, x2, x3, x4, x5 = torch.randn((32, 196, 256)), torch.randn((32, 196, 256)), torch.randn((32, 196)), torch.randn((32, 197)), torch.randn((32, 196))
            x1, x2, x3, x4, x5 = model(img.to(device))
            loss = loss_fn(x1, x2, x3, x4, x5)
            print(loss)
