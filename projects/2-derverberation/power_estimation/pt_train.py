import os
from params import *
from pt_dervb_net import DervbNet
from pt_dervb_net_single_channel import DervbNet

from pt_trainer import DervbTrainer

# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2'
scratch = True
if not scratch:
    model_name = 'b32-best-32.pt.tar'
    model_path = sum_dir + '/model_b-32_lr-0.001/' + model_name


def train():
    import torch
    import numpy as np
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    gpuids = tuple(map(int, gpu_id.split(",")))
    print('>> Initialzing nnet...')
    nnet = DervbNet()
    trainer = DervbTrainer(
        nnet,
        gpuid=gpuids,
        save_model_dir=model_save_dir,
        load_model_path=None if scratch else model_path,
        optimizer_kwargs={
            "lr": lr,
            "weight_decay": weight_decay
        })
    from dataset_tools.tasnet_dataset import TasnetDataset
    from torch.utils.data import DataLoader
    from dataset_tools.tasnet_data_loader import collate_fn
    train_dataset = TasnetDataset(training_path, mode='train_pretrain')
    validation_dataset = TasnetDataset(validation_path, mode='val')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=8, prefetch_factor=4, drop_last=True)
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0, drop_last=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    print('>> Preparing nnet...')
    trainer.run(train_dataloader, validation_dataloader, num_epochs=max_epoch)


if __name__ == '__main__':
    if not os.path.exists(sum_dir):
        os.mkdir(sum_dir)
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    train()
