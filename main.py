import torch
from torch.utils.tensorboard import SummaryWriter
from dataset.plant_pathology import PlantClassification
from model.dicls import DiCLS
from model.utils.configs import Config
from model.utils.evaluator import evaluate
from model.utils.logger import log_tensorboard, log_detail, setup_logger
from argparse import ArgumentParser
import torch.optim as optim
from torch.nn import functional as F
from dataset.utils import label2caption, get_caption
from dataset.dataloader import create_dataloader
from tqdm import tqdm
import os
import logging
import numpy as np
import time

PATH = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(PATH, 'logs')

def train(cfg, model, dataloader, idx2label, start_time=None, writer=None, step=None):
    results = {}

    mean_loss = None
    for i, data in enumerate(dataloader):
        step[0] += 1

        imgs, labels, plains = data
        bz = imgs.size(0)
        
        #ids = torch.nonzero(labels == 1).squeeze().tolist() 
        #infos = {}
        #all_labels = ",".join([label for label in idx2label.values()])
        #captions = get_caption(ids, idx2label, infos)
        #captions = ["All:" + all_labels + ",Detect:" + captions[i] + ",Description: green, leaf, disease" for i in range(bz)]
        #captions = ["Detect:" + captions[i] for i in range(bz)]
        #captions = [all_labels + "[SEP] Description: green, leaf, disease" for i in range(bz)]
    
        # with torch.no_grad():
        #     tokens = model.tokenizer(
        #         captions,
        #         return_tensors="pt",
        #         truncation=True,
        #         padding="max_length",
        #         max_length=cfg.tokenizer_max_length,
        #     )

        # TODO: move this to collect_fn
        suffix = 'Description: green leaf with disease'
        #suffix = None
        tokens, targets = get_caption(model.tokenizer, idx2label, 
                                      plains, None, suffix, cfg.tokenizer_max_length)
        targets = torch.Tensor(targets)
        #print([idx2label[i] for i, k in enumerate(labels[0]) if k])
        #vocab = {v : k for k, v in model.tokenizer.get_vocab().items()}
        #sent = [vocab[int(token.numpy())] for token in tokens['input_ids'][0]]

        inputs = (imgs.to(device), tokens.to(device))
        word_targets = targets.float().to(device)
        targets = labels.float().to(device)
        
        #print(sent, labels[0])

        all_loss = model(inputs, targets, word_targets)
        
        loss = all_loss['total']

        if mean_loss is None:
            mean_loss = {name : loss.item() for name, loss in all_loss.items()}
        else:
            mean_loss = {name : loss.item() + mean_loss[name] for name, loss in all_loss.items()}

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scheduler.step()

        with torch.no_grad():
            if (i + 1) % cfg.loss_period == 0:
                logging.info(f'Epoch [{epoch+1}/{num_epochs}], Progress [{i}/{data_len}], Loss: {loss.item() : .2f}, Learning Rate: {optimizer.param_groups[0]["lr"]}')

        if cfg.tensorboard:
            log_tensorboard({name : loss.item() for name, loss in all_loss.items()}, writer, step[0])
            
            # if (i + 1) % cfg.detail_period == 0:
            #     logging.info("Detail:", [(k, str(v.detach().item())[:5]) for k, v in all_loss.items()])

    mean_loss = {name : loss / len(dataloader) for name, loss in mean_loss.items()}
    if start_time is not None:
        results.update({"time_cost": time.time() - start_time})
    results.update(mean_loss)
    
    return results
            

def test(cfg, model, dataloader, idx2label, test=False):

    all_logits, all_targets = [], []
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader, 'Testing: ')):
            imgs, labels, plains = data
            bz = imgs.size(0)
            
            # ids = torch.nonzero(labels == 1).squeeze().tolist()
            
            #captions = get_caption(ids, idx2label, infos)
            #captions = [captions[i] for i in range(bz)]
            #captions = ["All: " + ",".join([label for label in idx2label.values()]) + "[SEP] Description: green,leaf,disease"] * bz

            suffix = 'Description: green leaf with disease'
            #suffix = None
            tokens, targets = get_caption(model.tokenizer, idx2label, 
                                          plains, None, suffix, cfg.tokenizer_max_length)
            
            # tokens = model.tokenizer(
            #     captions,
            #     return_tensors="pt",
            #     truncation=True,
            #     padding="max_length",
            #     max_length=cfg.tokenizer_max_length,
            # )

            targets = torch.Tensor(targets)

            inputs = (imgs.to(device), tokens.to(device))
            labels = labels.long()

            outputs, logits, probs = model(inputs)
            #print(labels[0], probs[0])
            #predictions = (probs > cfg.pred_threshold).long()

            all_logits.append(probs.detach().cpu().numpy())
            all_targets.append(labels.detach().numpy())
        
        #print([x.shape for x in all_logits])
        all_logits = np.vstack(all_logits)
        all_targets = np.vstack(all_targets)

        #print(all_logits[0], all_targets[0])

        results = evaluate(all_logits, all_targets, cfg.num_class, 
                           cfg.pred_threshold, **cfg.eval.dict())
        
        if test:
            results.update(outputs)
        
        return results


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--batch_size', default=32, help='batchsize to train.')
    parser.add_argument('--epochs', default=25, help='epochs to run.')
    parser.add_argument('--config', default=None, help='model config to create model.') # TODO
    parser.add_argument('--early_stop', default='3', help='enable early stop with \'early_stop\' epochs.') # TODO
    parser.add_argument('--path', default='data', help='select your dataset path.')
    parser.add_argument('--dataset_type', default='PlantCLS', help='select your dataset type.')
    parser.add_argument('--device', default='cuda:0', help='select your device.')

    parser.add_argument('--lr', default='0.00001', help='learning rate.')
    # parser.add_argument('--lr_scheduler', default='cos', help='select your device.')
    parser.add_argument('--loss_period', default=5, help='step intervals to print total loss & lr.')
    parser.add_argument('--test_period', default=1, help='epoch intervals to print detail test results.')
    parser.add_argument('--save_period', default=5, help='epoch intervals to save model ckpt.')

    parser.add_argument('--keep', action='store_true', help='start training from last checkpoint.')
    parser.add_argument('--visualize', action='store_true', help='visualize training results.') # TODO
    parser.add_argument('--exp_name', default='exp', help='experiment name to save logs.')
    parser.add_argument('--tensorboard', default=True, help='use tensorboard.')

    parser.add_argument('--test', action='store_true', help='only test model.')
    parser.add_argument('--weight', default=None, help='test model path/last checkpoint path.')
    parser.add_argument('--log_level', default='DEBUG', help='logging\'s log level.')
    
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    torch.backends.cudnn.enable = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(3407)

    args = parser.parse_args()
    cfg = Config()
    cfg.merge_from_config(args)

    log_path, weight_name = setup_logger(cfg, LOG_DIR, 'test' if cfg.test else 
                                         'train', cfg.exp_name, cfg.log_level, cfg.keep)

    start_time = time.time()
    st_epoch = 0
    weight_path = os.path.join(log_path, 'weights')

    writer = None
    if cfg.tensorboard:
        writer = SummaryWriter(log_path)

    device = torch.device(cfg.device)
    lr = float(cfg.lr)
    num_epochs = int(cfg.epochs)
    bz = int(cfg.batch_size)

    model = DiCLS(cfg).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    steps, max_acc = [0], 0.

    if (cfg.keep and (os.path.exists(weight_name) or cfg.weight)) or cfg.weight and cfg.test:
        weight_name = cfg.weight if cfg.weight else weight_name
        logging.info(f'loading checkpoint from {weight_name}...')

        checkpoint = torch.load(weight_name)
        st_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        max_acc = checkpoint['max_acc']

    train_dataloader = create_dataloader('PlantCLS', cfg.path, bz, True, dataset_type='train', use_npz=True, label_smooth=0.02, mixup=0.1)
    val_dataloader = create_dataloader('PlantCLS', cfg.path, bz, True, dataset_type='val', use_npz=True)
    if cfg.test:
        test_dataloader = create_dataloader('PlantCLS', cfg.path, bz, True, dataset_type='test', use_npz=True)

    if cfg.use_ori_classnames:
        idx2label = train_dataloader.dataset.idx2label
    else:
        idx2label = {i : 'Class' + str(v) for i, v in enumerate(range(6))}
    data_len = len(train_dataloader)

    if not cfg.test:

        for epoch in range(st_epoch, num_epochs):
            model.train()
            train_results = train(cfg, model, train_dataloader, idx2label, start_time, writer=writer, step=steps)

            train_results.update({"epoch": epoch + 1})
            log_detail(train_results, 'Train Result')
            
            if (epoch + 1) % cfg.test_period == 0:
                model.eval()
                test_results = test(cfg, model, val_dataloader, idx2label)
                if cfg.tensorboard:
                    log_tensorboard(test_results, writer, epoch)
                test_results.update({"epoch": epoch + 1})
                log_detail(test_results, 'Test Result')
            
            if (epoch + 1) % cfg.save_period == 0:
                model_dict = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "max_acc": max_acc
                }

                model_path = os.path.join(weight_path, f'model_{epoch}.pt')
                logging.info(f'Saving model to {model_path}...')
                torch.save(model_dict, model_path)
                if max_acc < test_results.get("accuracy"):
                    max_acc = test_results.get("accuracy")
                    
                    best_model_path = os.path.join(log_path, 'best_model.pt')
                    logging.info(f'Saving best model to {best_model_path}')
                    torch.save(model_dict, best_model_path)
    else:
        model.eval()
        test_results = test(cfg, model, test_dataloader, idx2label, True)
        log_detail(test_results, 'Test Result')

    if writer:
        writer.close()