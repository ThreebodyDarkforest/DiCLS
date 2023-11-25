from torch.utils.tensorboard import SummaryWriter
import logging
from datetime import datetime
import os
from typing import Dict
from .evaluator import SCALAR, CURVE
import matplotlib.pyplot as plt
from model.utils.configs import save_config

def setup_logger(cfg, log_dir='logs', task='train', 
                 exp_name=None, log_level='DEBUG', keep=False):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if exp_name is None:
        exp_name = f"{task}_{datetime.now().strftime('%m_%d-%H_%M')}"
    elif not keep:
        dirpath = os.path.join(log_dir, exp_name)
        if os.path.exists(dirpath):
            match = [folder for folder in os.listdir(log_dir) if 
                                folder.startswith(exp_name + '_') and 
                                folder[len(exp_name) + 1:].isdigit()]
            
            if len(match) >= 1:
                last = max(int(folder.split('_')[-1]) for folder in match)
                exp_name += f'_{last + 1}'
            else:
                exp_name += '_1'

    log_format = '[%(asctime)s] - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format)

    log_path = os.path.join(log_dir, exp_name)
    log_filepath = os.path.join(log_path, 'log.txt')
    weight_path = os.path.join(log_path, 'weights')
    weight_name = os.path.join(weight_path, 'model')

    match = []
    if cfg.keep:
        match = [file for file in os.listdir(weight_path) if 
                file.startswith(os.path.basename(weight_name) + '_') and 
                file[len(os.path.basename(weight_name)) + 1:-3].isdigit()]

    if len(match) >= 1:
        last = max(int(file.split('_')[-1][:-3]) for file in match)
        weight_name += f'_{last}'
    else:
        weight_name += '_0'

    weight_name += '.pt'

    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)

    save_config(os.path.join(log_path, 'config.yaml'), cfg)

    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(log_format))

    logging.getLogger('').addHandler(file_handler)

    logging.info('Log file created at: %s', log_filepath)
    logging.info('Logging started at: %s', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    return log_path, weight_name

def log_tensorboard(results: dict, writer: SummaryWriter, step=None):
    num_classes = results.get('num_classes')

    for name, eval in results.items():
        
        if name not in CURVE:
            writer.add_scalar(name, eval, step)

        elif name == 'roc_curve':
            fig = plt.figure(figsize=(8, 6))
            fpr, tpr, roc_auc = eval
            for i in range(num_classes):
                plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
            plt.xlabel('FPR')
            plt.ylabel('TPR')
            plt.legend()

            writer.add_figure(name, fig, step)
            plt.cla()

        elif name == 'micro_roc':
            fig = plt.figure(figsize=(8, 6))
            fpr_micro, tpr_micro, roc_auc_micro = eval
            plt.plot(fpr_micro, tpr_micro, label=f'Micro-average (AUC = {roc_auc_micro:.2f})', linestyle='--', color='black')
            plt.xlabel('FPR')
            plt.ylabel('TPR')
            plt.legend()

            writer.add_figure(name, fig, step)
            plt.cla()
        
        if name == 'pr_curve':
            fig = plt.figure(figsize=(8, 6))
            precision, recall, thresholds = eval
            plt.plot(recall[1:], precision[1:], color='blue', lw=1, label='PR Curve')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.legend()
            
            writer.add_figure(name, fig, step)
            plt.cla()

def log_detail(results: dict, topic='Train Results'):
    print("=" * 25)
    print("{:<20}".format(topic))
    print("{:<15} {:<10}".format('Metric', 'Value'))
    print("=" * 25)

    for metric, value in results.items():
        if metric in CURVE: continue
        if isinstance(value, int):
            print("{:<15} {:d}".format(metric.capitalize(), value))
        elif isinstance(value, float):
            print("{:<15} {:.4f}".format(metric.capitalize(), value))
    print("=" * 25)