from __future__ import print_function
import os
import sys
import time
from typing import List

from torch.nn.utils.rnn import pad_sequence

from params import *
from itertools import permutations
from collections import defaultdict, OrderedDict
import threading
import queue
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
import logging
from data_tools_zt.utils.ark_run_tools import ArkRunTools
#import torch_complex.functional as FC
#from torch_complex import ComplexTensor


class MyThread(threading.Thread):
    def __init__(self, func, args=None, name=''):
        super(MyThread, self).__init__()
        self.name = name
        self.func = func
        self.args = args

    def run(self):
        if self.args is None:
            self.result = self.func()
        else:
            self.result = self.func(self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None


def get_logger(
        name,
        format_str="%(asctime)s [%(lineno)s - %(levelname)s ] %(message)s",
        date_format="%Y-%m-%d %H:%M:%S",
        file=False):
    """
    Get python logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # file or console
    handler = logging.StreamHandler() if not file else logging.FileHandler(
        name)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt=format_str, datefmt=date_format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def offload_data(data, device):
    def _cuda(obj):
        return obj.to(device) if isinstance(obj, torch.Tensor) else obj

    def offload(obj):
        if cuda:
            obj_cuda = _cuda(obj)
            if isinstance(obj_cuda, list):
                obj_cuda = list(map(_cuda, obj_cuda))
        else:
            obj_cuda = obj
        return obj_cuda

    new_data = dict()
    for key in data.keys():
        new_data[key] = offload(data[key])
    return new_data


class AverageMeter(object):
    """
    A simple average meter
    """

    def __init__(self):
        self.val = defaultdict(float)
        self.cnt = defaultdict(int)

    def reset(self):
        self.val.clear()
        self.cnt.clear()

    def add(self, key, value, cnt=1):
        self.val[key] += value * cnt
        self.cnt[key] += cnt

    def value(self, key):
        if self.cnt[key] == 0:
            return 0
        return self.val[key] / self.cnt[key]

    def sum(self, key):
        return self.val[key]

    def count(self, key):
        return self.cnt[key]


class SimpleTimer(object):
    """
    A simple timer
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.start = time.time()

    def elapsed(self):
        return time.time() - self.start


def check_gradient(model: torch.nn.Module) -> bool:
    for name, param in model.named_parameters():
        # print("param name is {}".format(name))
        if not torch.all(torch.isfinite(param.grad)):
            return False
    return True


class Trainer(object):
    def __init__(self,
                 nnet,
                 name=model_name,
                 save_model_dir=model_save_dir,
                 optimizer="adam",  # adam
                 gpuid=0,
                 optimizer_kwargs=None,
                 clip_norm=None,
                 min_lr=1e-8,
                 patience=3,
                 factor=0.5,
                 logging_period=log_display_step,
                 load_model_path=None):
        if not torch.cuda.is_available() and cuda:
            raise RuntimeError("CUDA device unavailable...exist")
        if not isinstance(gpuid, tuple):
            gpuid = (gpuid,)
        self.device = torch.device("cuda:{}".format(gpuid[0]))  # todo does it means it can be run on one device
        self.name = name
        self.gpuid = gpuid
        if save_model_dir and not os.path.exists(save_model_dir):
            os.makedirs(save_model_dir)
        self.save_model_dir = save_model_dir
        self.logger = get_logger(
            os.path.join(save_model_dir, "trainer-" + model_name + ".log"), file=True)

        self.clip_norm = clip_norm
        self.logging_period = logging_period
        self.cur_epoch = 0  # zero based
        self.forward_count = 0 
        self.accum_grad = accum_grad
        self.nnet = nnet
        if load_model_path:
            if not os.path.exists(load_model_path):
                raise FileNotFoundError(
                    "Could not find load_model_path checkpoint: {}".format(load_model_path))
            cpt = torch.load(load_model_path, map_location="cpu")
            self.cur_epoch = cpt["epoch"]
            #import pdb; pdb.set_trace()
            self.logger.info("Resume from checkpoint {}: epoch {:d}".format(
                load_model_path, self.cur_epoch))
            # load nnet
            nnet.load_state_dict(cpt["model_state_dict"])
            if cuda and torch.cuda.is_available():
                self.nnet = nnet.to(self.device)
            # import pdb; pdb.set_trace()
            # finetune
            self.optimizer = self.create_optimizer(optimizer, optimizer_kwargs)
            # self.optimizer = self.create_optimizer(optimizer, optimizer_kwargs, state=cpt["optim_state_dict"])
        else:
            if cuda and torch.cuda.is_available():
                self.nnet = nnet.to(self.device)
            self.optimizer = self.create_optimizer(optimizer, optimizer_kwargs)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=factor,
            patience=patience,
            min_lr=min_lr,
            verbose=True)
        self.num_params = sum(
            [param.nelement() for param in nnet.parameters()]) / 10.0 ** 6

        # logging
        self.logger.info("Model summary:\n{}".format(nnet))
        self.logger.info("Loading model to GPUs:{}, #param: {:.2f}M".format(
            gpuid, self.num_params))
        if clip_norm:
            self.logger.info(
                "Gradient clipping by {}, default L2".format(clip_norm))
        self.data_queue = queue.Queue()

    def save_checkpoint(self, best=True):
        cpt = {
            "epoch": self.cur_epoch,
            "model_state_dict": self.nnet.state_dict(),
            "optim_state_dict": self.optimizer.state_dict()
        }
        torch.save(
            cpt,
            os.path.join(self.save_model_dir,
                         "{0}-{1}-{2}.pt.tar".format(self.name, "best" if best else "last", self.cur_epoch)))

    def create_optimizer(self, optimizer, kwargs, state=None):
        supported_optimizer = {
            "sgd": torch.optim.SGD,  # momentum, weight_decay, lr
            "rmsprop": torch.optim.RMSprop,  # momentum, weight_decay, lr
            "adam": torch.optim.Adam,  # weight_decay, lr
            "adadelta": torch.optim.Adadelta,  # weight_decay, lr
            "adagrad": torch.optim.Adagrad,  # lr, lr_decay, weight_decay
            "adamax": torch.optim.Adamax  # lr, weight_decay
            # ...
        }
        if optimizer not in supported_optimizer:
            raise ValueError("Now only support optimizer {}".format(optimizer))
        opt = supported_optimizer[optimizer](self.nnet.parameters(), **kwargs)
        self.logger.info("Create optimizer {0}: {1}".format(optimizer, kwargs))
        if state is not None:
            opt.load_state_dict(state)
            self.logger.info("Load optimizer state dict from checkpoint")
        return opt

    def compute_loss(self, data):
        raise NotImplementedError

    def train(self, data_loader):
        self.nnet.train()
        stats = AverageMeter()
        timer = SimpleTimer()
        
        for data in data_loader:
            train_task = MyThread(self.compute_loss, args=data)
        
            train_task.start()
            train_task.join() 

            loss = 0
            loss_mse, loss_sisnr, loss_lfb = train_task.get_result()
            # loss = loss_mse
            # loss = loss_lfb
            # loss = loss_mse + loss_lfb
            if 'stft' in loss_type:
                loss += loss_mse
            if 'sisnr' in loss_type:
                loss += loss_sisnr
            if 'lfb' in loss_type:
                loss += loss_lfb
            stats.add("loss", loss.item())
            loss.backward()   
            progress = stats.count("loss")  # batch cnt
            ## accum_grad
            self.forward_count += 1
            if self.forward_count != self.accum_grad:
                # print(f"self.forward_count: {self.forward_count}, self.accum_grad: {self.accum_grad}")
                continue
            if not progress % (self.accum_grad*self.logging_period):
                self.logger.info("Process {:d}-th batch and its stft mse: {:.3f}, sisnr: {:.3f}, lfb mse: {:.3f}, back loss is : {:.3f}".format(progress, loss_mse.item(), loss_sisnr.item(), loss_lfb.item(), loss.item()))
            self.forward_count = 0
            
            if self.clip_norm:
                clip_grad_norm_(self.nnet.parameters(), self.clip_norm)

            self.optimizer.step()  # update model parameters
            self.optimizer.zero_grad()  # set the gradient of all parameters to zero

        return stats.value("loss"), stats.count("loss"), timer.elapsed()

    def evaluate(self, data_loader):
        self.nnet.eval()
        stats = AverageMeter()
        timer = SimpleTimer()

        with torch.no_grad():
            for data in data_loader:
                loss = 0
                loss_mse, loss_sisnr, loss_lfb  = self.compute_loss(data)
                # loss = loss_mse
                # loss = loss_lfb
                # loss = loss_mse + loss_lfb
                if 'stft' in loss_type:
                    loss += loss_mse
                if 'sisnr' in loss_type:
                    loss += loss_sisnr
                if 'lfb' in loss_type:
                    loss += loss_lfb
                this_batch_size = data[list(data.keys())[0]].size()[0]
                print(f"this_batch_size:{this_batch_size}")
                stats.add("loss", value=loss.item(), cnt=this_batch_size)
                stats.add("sisnr", value=loss_sisnr.item(), cnt=this_batch_size)
                stats.add("lfb", value=loss_lfb.item(), cnt=this_batch_size)
                stats.add("stft", value=loss_mse.item(), cnt=this_batch_size)
        # return first loss as the validation loss to guide lr
        return stats.value('loss'), stats.count('loss'), timer.elapsed(), stats.value('sisnr'), stats.value('lfb'), stats.value('stft')

    def run(self, train_loader, dev_loader, num_epochs=100):
        stats = dict()
        print('>> Check if save is OK...', end='')
        self.save_checkpoint(best=False)
        print('done\n>> Scratch evaluating...', end='')
        best_loss = 10
        self.logger.info("START FROM EPOCH {:d}".format(self.cur_epoch))
        while self.cur_epoch < num_epochs:
            stats["title"] = "Loss(time/N, lr={:.3e}) - Epoch {:2d}:".format(self.optimizer.param_groups[0]["lr"],
                                                                             self.cur_epoch + 1)
            tr_loss, tr_batch, tr_cost = self.train(train_loader)
            print('End epoch {:d}, train loss: {:.5f}'.format(self.cur_epoch, tr_loss))
            stats["tr"] = "train = {:+.5f}({:.2f}s/{:d})".format(tr_loss, tr_cost, tr_batch)
            cv_loss, cv_batch, cv_cost, cv_sisnr_loss, cv_lfb_loss, cv_stft_loss = self.evaluate(dev_loader)
            stats["cv"] = "dev_loss = {:+.5f}, ({:.2f}s/{:d}), sisnr= {:.5f}, lfb= {:.5f}, stft= {:.5f}".format(cv_loss,
                                                                                                  cv_cost,
                                                                                                  cv_batch,
                                                                                                  cv_sisnr_loss,
                                                                                                  cv_lfb_loss, 
                                                                                                  cv_stft_loss)
            stats["scheduler"] = ""

            # update current epoch num after loss calculated
            self.cur_epoch += 1

            if cv_loss > best_loss:
                stats["scheduler"] = "| no impr, best = {:.5f}".format(best_loss)
            else:
                best_loss = cv_loss
                self.save_checkpoint(best=True)
            self.logger.info("{title} {tr} | {cv} {scheduler}".format(**stats))
            # select metrics (such as cv_loss1) and schedule here to update lr
            self.scheduler.step(cv_loss)
            # flush scheduler info
            sys.stdout.flush()
            # save checkpoint of current epoch
            self.save_checkpoint(best=False)

        self.logger.info("Training for {:d} epoches done!".format(num_epochs))


class DervbTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(DervbTrainer, self).__init__(*args, **kwargs)

    def sisnri(self, x, s, m):
        """
        Arguments:
        x: separated signal, BS x S
        s: reference signal, BS x S
        m: mixture signal, BS x S
        Return:
        sisnri: N tensor
        """

        sisnr = self.sisnr(x, s)
        sisnr_ori = self.sisnr(m, s)
        return sisnr - sisnr_ori

    # def lfb_mse(self, x, s, eps=1e-8):
    #     """
    #     x: estimated spectral, (B, C, F, T)
    #     s: reference spectral, (B, C, F, T) B x F x T
    #     return: log fbank MSE: B tensor
    #     """
    #     import pdb; pdb.set_trace()
    #     if x.shape != s.shape:
    #         if x.shape[-1] > s.shape[-1]:
    #             x = x[:,:,:s.shape[-1]]
    #         else:
    #             s = s[:,:,:x.shape[-1]]

    #     lfb_x = self.nnet.lfb(x.permute((0,2,1)), self.nnet.epsilon)
    #     lfb_s = self.nnet.lfb(s.permute((0,2,1)), self.nnet.epsilon)
    #     t = torch.sum((lfb_x - lfb_s) ** 2, dim=-1)
    #     t = torch.sum(t, dim=-1)
    #     return t

    def lfb_mse(self, x, s, lens):
        """
        x: estimated spectral, (B, C, F, T)
        s: reference spectral, (B, C, F, T)
        lens: (B,)
        return: tensor one value
        """
        # import pdb; pdb.set_trace()
        if x.shape != s.shape:
            if x.shape[-1] > s.shape[-1]:
                x = x[:,:,:s.shape[-1]]
            else:
                s = s[:,:,:x.shape[-1]]
        # (B, C, F, T) -> (B, C, T, F) -> (B, C, T, F'': 120)
        lfb_x = self.nnet.lfb(x.transpose(-1,-2), self.nnet.epsilon)
        lfb_s = self.nnet.lfb(s.transpose(-1,-2), self.nnet.epsilon)
        t = torch.sum((lfb_x - lfb_s) ** 2, dim=-1)
        t = torch.sum(t, dim=-1)
        # t_new = self.mse_loss(lfb_x.transpose(-1, -2), lfb_s.transpose(-1, -2), lens)
        return t

    def sisnr(self, x, s, eps=1e-8):
        """
        Arguments:
        x: separated signal, (B, C, t)
        s: reference signal, (B, C, t)
        Return:
        sisnr: (B, C)
        """
        # Note, here we only foucs first channel

        def l2norm(mat, keepdim=False):
            return torch.norm(mat, dim=-1, keepdim=keepdim)
 
        if x.dim() == 1:
            x = x.unsqueeze(0)
            if s.dim() == 1:
                s = s.unsqueeze(0)
        if x.shape != s.shape:
            raise RuntimeError(
                "Dimension mismatch when calculate si-snr, {} vs {}".format(
                    x.shape, s.shape))
        # x_zm, s_zm: (B, C, t)
        # todo 根据实际音频的长度，计算有效的均值，从而得到x_zm 和 s_zm
        x_zm = x - torch.mean(x, dim=-1, keepdim=True)
        s_zm = s - torch.mean(s, dim=-1, keepdim=True)
        # 每个部分都乘以了s_zm，s_zm的无效部分都是0，所以t的计算，不受无效部分的影响
        # 主要是x_zm和s_zm的计算不准, 但无效部分的取值很小，所以x_zm和s_zm整体计算不受影响。
        # t: (B, C, t)
        t = torch.sum(x_zm * s_zm, dim=-1,keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True) ** 2 + eps)
        # return: (B, C)
        return 20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))

    def mse_loss(self, ipt, target, n_frames):
        """
        Calculate the MSE loss for variable length dataset
        ipt: (B, C, F, T)
        target: (B, C, F, T)
        n_frames: (B,)
        return: tensor one value
        """
        E = 1e-7
        # import pdb; pdb.set_trace() 
        with torch.no_grad():
            masks = []
            for n_frame in n_frames:
                # the mask shape is (T, F) 
                masks.append(torch.ones((n_frame, target.size(-2)), dtype=torch.float32))  
            # binary_mask: 有实际数值的地方填充的1，为了对其的位置填充的0(代表非实际数值，不参与loss计算)，且数值都是float类型
            # B * [T, F] -> (B, T, F)
            binary_mask = pad_sequence(masks, batch_first=True).to(ipt.device)
            # (B, T, F) -> (B, 1, T, F) -> (B, 1, F, T) # one mask for all channel
            binary_mask = binary_mask.unsqueeze(1).transpose(-1, -2)
        # masked_ipt: (B, C, F, T)
        masked_ipt = ipt * binary_mask
        # masked_target: (B, C, F, T)
        masked_target = target * binary_mask
        # import pdb; pdb.set_trace()
        return ((masked_ipt - masked_target) ** 2).sum() / (n_frames.sum() * ipt.shape[1] + E)

    def compute_loss(self, data_all):
        data = offload_data(data_all, self.device)
        
        # we need to divie ArkRunTools.C when using ark_scp
        with torch.no_grad():
            data["mix"] = data["mix"]/ArkRunTools.C
            # print(f"new_wav_scp: {new_wav_scp}")
            # data["ref"] = data["ref"]/ArkRunTools.C
            data["ref"] = data["ref"]
        # ests: [(B, C, t)],  mag: [est: (B, C, F, T), ref: (B, C, F, T)]
        ests, mag, comps = torch.nn.parallel.data_parallel(self.nnet,
                                                                [data["mix"],
                                                                 data["lip_video"],
                                                                 data["ref"],
                                                                 data["ori_wav_len"]],
                                                               device_ids=self.gpuid)
        estspec = [mag[0]]
        refspec = mag[1]
        real_ref = comps[0]  # (B, C, F, T)
        imag_ref = comps[1]  # (B, C, F, T)
        real_est = comps[2]  # (B, C, F, T)
        imag_est = comps[3]  # (B, C, F, T)
        # for scale
        # import pdb; pdb.set_trace()
        # loss_mse = (torch.sum((real_ref-real_est) ** 2) + torch.sum((imag_ref-imag_est) ** 2))/(real_ref.shape[-1] * real_ref.shape[0] * * real_ref.shape[1])
        lens = data["ori_wav_len"]
        loss_mse = (self.mse_loss(real_est, real_ref, lens) + self.mse_loss(imag_est, imag_ref, lens)) * 10
        # import pdb; pdb.set_trace()
        # print(f"loss mse {loss_mse}, loss_mse_new {loss_mse_new}")

        # ######## sisnr and lfb_mse ##########
        # import pdb; pdb.set_trace()
        num_spks = out_spk # 1
        # import pdb; pdb.set_trace()

        def sisnr_loss(permute):
            return sum([self.sisnr(ests[s], data["ref"]) for s, t in enumerate(permute)]) / len(permute)

        def lfb_loss(permute):
            return sum([self.lfb_mse(estspec[s], refspec, lens) for s, t in enumerate(permute)]) / len(permute)
        
        # import pdb; pdb.set_trace()
        spec_mat = torch.stack([lfb_loss(p) for p in permutations(range(num_spks))])
        max_perutt_spec, _ = torch.max(spec_mat, dim=0)
        loss_lfb = torch.mean(max_perutt_spec) * 1e-4
        # loss_lfb = torch.mean(max_perutt_spec)

        sisnr_mat = torch.stack([sisnr_loss(p) for p in permutations(range(num_spks))])
        max_perutt, _ = torch.max(sisnr_mat, dim=0)
        loss_sisnr = -torch.sum(max_perutt) / (data["mix"].size(0) * data["mix"].size(1))
        
        print(f"loss mse {loss_mse}, loss_sisnr : {loss_sisnr}, loss_lfb: {loss_lfb}")

        return loss_mse, loss_sisnr, loss_lfb
