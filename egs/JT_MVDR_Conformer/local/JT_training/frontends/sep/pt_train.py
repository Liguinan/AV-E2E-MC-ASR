from params import *
from pt_avnet import ConvTasNet
from data_generator import CHDataLoader
import os
from pt_trainer import SiSnrTrainer

# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'
scratch_or_resume = True
if not scratch_or_resume:
    model_subpath = 'TF-LPS_IPD_AF_LIP-model.pt.tar'
    model_path = sum_dir + '/model_tfmasking/'+model_subpath

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
    nnet = ConvTasNet(norm=norm,
                      out_spk=out_spk,
                      non_linear=activation_function,
                      causal=causal,
                      cosIPD=cosIPD,
                      input_features=input_features,
                      spk_fea_dim=speaker_feature_dim,
		      )

    trainer = SiSnrTrainer(
        nnet,
        gpuid=gpuids,
        clip_norm=10,
        save_model_dir=model_save_dir,
        load_model_path=None if scratch_or_resume else model_path,
        optimizer_kwargs={
            "lr": lr,
            "weight_decay": lr_decay
        })
    from dataset_tools.tasnet_dataset import TasnetDataset
    from torch.utils.data import DataLoader
    from dataset_tools.tasnet_data_loader import collate_fn
    # training_path = "/project_bdda7/bdda/tzhong/projects/Speech-Enhancement-Compression/data/train/ark_pickle/train_pretrain_32_rev1_le6_ark.pkl"
    # validation_path = "/project_bdda7/bdda/tzhong/projects/Speech-Enhancement-Compression/data/validation/ark_pickle/val_rev4_le6_ark.pkl"
    train_dataset = TasnetDataset(training_path)
    validation_dataset = TasnetDataset(validation_path)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=8, prefetch_factor=4, drop_last=True)
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0, drop_last=True)
    # tr_loader = CHDataLoader('tr', batch_size)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    # cv_loader = CHDataLoader('cv', batch_size)

    print('>> Preparing nnet...')
    trainer.run(train_dataloader, validation_dataloader, num_epochs=max_epoch, warm_up_epochs=5)



if __name__ == '__main__':
    if not os.path.exists(sum_dir):
        os.mkdir(sum_dir)
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    train()
