from __future__ import print_function
import os
import sys
import time
from params import *
from itertools import permutations
from collections import defaultdict
import threading
import queue
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
import logging
from data_tools_zt.utils.ark_run_tools import ArkRunTools
#from pt_log_fbank import LFB
from torch import autograd
class MyThread(threading.Thread):
    def __init__(self, func, args=None, name=''):
        super(MyThread, self).__init__()
        self.name = name
        self.func = func
        self.args = args


    def run(self):
        if self.args is None:
            self.result =self.func()
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


class Trainer(object):
    def __init__(self,
                 nnet,
                 name=model_name,
                 out_spk=out_spk,
                 save_model_dir=model_save_dir,
                 optimizer="adam",
                 gpuid=0,
                 optimizer_kwargs=None,
                 clip_norm=10,
                 min_lr=1e-8,
                 patience=3,
                 factor=0.5,
                 logging_period=log_display_step,
                 load_model_path=None):
        if not torch.cuda.is_available() and cuda:
            raise RuntimeError("CUDA device unavailable...exist")
        if not isinstance(gpuid, tuple):
            gpuid = (gpuid,)
        self.device = torch.device("cuda:{}".format(gpuid[0]))
        self.name = name
        self.gpuid = gpuid
        if save_model_dir and not os.path.exists(save_model_dir):
            os.makedirs(save_model_dir)
        self.save_model_dir = save_model_dir
        self.out_spk = out_spk
        self.logger = get_logger(
            os.path.join(save_model_dir, "trainer-moplast-" + model_name + ".log"), file=True)

        self.clip_norm = clip_norm
        self.logging_period = logging_period
        self.cur_epoch = 0  # zero based
        self.nnet = nnet
        if load_model_path:
            if not os.path.exists(load_model_path):
                raise FileNotFoundError(
                    "Could not find load_model_path checkpoint: {}".format(load_model_path))
            cpt = torch.load(load_model_path, map_location="cpu")

            self.cur_epoch = cpt["epoch"]
            my_dict = nnet.state_dict()
            pretrained_dict = {k: v for k, v in cpt["model_state_dict"].items() if k in my_dict}
            my_dict.update(pretrained_dict)
            nnet.load_state_dict(my_dict)
            self.logger.info("Resume from checkpoint {}: epoch {:d}".format(
                load_model_path, self.cur_epoch))
            # load nnet
            #nnet.load_state_dict(cpt["model_state_dict"])
            if cuda and torch.cuda.is_available():
                self.nnet = nnet.to(self.device)
            # import pdb; pdb.set_trace()
            if resume:
                print("resume")
                self.optimizer = self.create_optimizer(
                    optimizer, optimizer_kwargs, state=cpt["optim_state_dict"])
            else:
                self.optimizer = self.create_optimizer(optimizer, optimizer_kwargs)

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
        self.logger.info("Model input features: {}".format(input_features))
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
        # import pdb; pdb.set_trace()
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

    def orthogonal_loss(self, regular_term=1e-5):
	#loss = self.nnet.av_fusion_layer.orthogonal_loss
	
	#loss *= regular_term
	
        param_list = self.nnet.named_parameters()
        w = self.nnet.av_fusion_layer.parallel_linear_transform.view(factor, -1)

        loss = torch.nn.functional.mse_loss(torch.matmul(w, w.transpose(1,0)), self.nnet.av_fusion_layer.I, reduction='mean')
#        for i in param_list:
#            if i[0].find("av_fusion_layer.parallel_linear_transform") != -1:
#                w_mat = i[1].view(factor, -1)
		#print(w_mat.shape)
 #               loss += torch.nn.functional.mse_loss(torch.matmul(w_mat, w_mat.transpose(1, 0)), self.nnet.I,
 #                                            reduce=True, size_average=False)
        loss *= regular_term
	
        return loss

    def regular_loss(self, regular_term=5e-3, p=2):
        param_list = self.nnet.named_parameters()
        loss = 0
	
        for i in param_list:
            #if i[0].find("av_fusion_layer.conv1d.weight") != -1 or 
            if i[0].find("conv1x1_1.weight") != -1: 
                reg = torch.norm(i[1], p=p)
                loss += reg
        loss *= regular_term
        return loss

    def train(self, data_loader):
        start_update = time.time()
        loss_type = ['LFB','SISNR']
        self.nnet.train()
        stats = AverageMeter()
        timer = SimpleTimer()
        flag = False
        # import pdb; pdb.set_trace()
        # path='/users/bdda/gnli/projects/bdda7_projects/TASLP-22/Front-End/separation/summary_0lipemb-LPS+IPD+AF-SISNR+SISNR_NOISE/model_mvdr-AF_premasking=False-Batch_Size=32-Dloading=1e-5-check-forward-time-dl'
        for data in data_loader:
            # start1 = time.time()
            threads = []
            self.optimizer.zero_grad()
            train_task = MyThread(self.compute_loss, args=(data))
	
            train_task.start()
            train_task.join()
	
            #loss_sisnr, loss_lfb, loss_sisnr_noise = train_task.get_result()
            loss_sisnr, loss_lfb = train_task.get_result()
            # end1 = time.time()
            # with open(f"{path}/front_forward.log", 'a') as fr:
            #     fr.write(f"{end1 - start1}\n")
            # print(f"front_forward: {end1 - start1}")
            
            #reg_loss = self.regular_loss()
            #orthogonal_loss = self.orthogonal_loss()
	
            all_loss = 0
            #if 'LFB' in loss_type:
            #    all_loss += loss_lfb
            if 'SISNR' in loss_type:
                all_loss += loss_sisnr 
                #all_loss = loss1 + loss2# + orthogonal_loss #+ reg_loss
            #if 'SISNR_NOISE' in loss_type:
            #    all_loss += loss_sisnr_noise
            #import pdb; pdb.set_trace()
            stats.add("loss", all_loss.item())
            # with autograd.detect_anomaly():
            # start2 = time.time()
            all_loss.backward()
            #print(loss, flag)
            #import pdb;pdb.set_trace()
            progress = stats.count("loss")
            if not progress % self.logging_period:
                self.logger.info("Processed {:d} batches, loss sisnr{:.2f}, loss lfb {:.2f}, loss sisnr_noise {:.2f} ...".format(progress, loss_sisnr.item() if 'SISNR' in loss_type else 0, loss_lfb.item() if 'LFB' in loss_type else 0, loss_sisnr_noise.item() if 'SISNR_NOISE' in loss_type else 0))#, orthogonal_loss.item()))

            if self.clip_norm:
                clip_grad_norm_(self.nnet.parameters(), self.clip_norm)
            self.optimizer.step()
            # end2 = time.time()
            # with open(f"{path}/front_back.log", 'a') as fr:
            #     fr.write(f"{end2 - start2}\n")
            # print(f"front_forward: {end2 - start2}")
            
        end_update1 = time.time()
        # with open(f"{path}/all_forward_back.log", 'a') as fr:
        #         fr.write(f"{end_update1 - start_update}\n")
        # print(f"all_forward_back: {end_update1 - start_update}")

        return stats.value("loss"), stats.count("loss"), timer.elapsed()

    def evaluate(self, data_loader):
        loss_type = ['LFB','SISNR']
        self.nnet.eval()
        stats = AverageMeter()
        timer = SimpleTimer()

        with torch.no_grad():
            for data in data_loader:
                loss_sisnr, loss_lfb = self.compute_loss(data)
                all_loss = 0
                #if 'LFB' in loss_type:
                #    all_loss += loss_lfb
                if 'SISNR' in loss_type:
                    all_loss += loss_sisnr
                #if 'SISNR_NOISE' in loss_type:
                #    all_loss += loss_sisnr_noise
                this_batch_size = data[list(data.keys())[0]].size()[0]
                # import pdb; pdb.set_trace()
                print(f"this_batch_size:{this_batch_size}")

                stats.add("loss", value=all_loss.item(), cnt=this_batch_size)
                # stats.add("loss", all_loss.item())
            
        return stats.value("loss"), stats.count("loss"), timer.elapsed()

    def run(self, train_loader, dev_loader, num_epochs=50, warm_up_epochs=20):
        # avoid alloc memory from gpu0
        # with torch.cuda.device(self.gpuid[0]):
        stats = dict()
        print('>> Check if save is OK...', end='')
        self.save_checkpoint(best=False)
        print('done\n>> Scratch evaluating...', end='')
        # best_loss, _, _ = self.evaluate(dev_loader)
        best_loss = 10
        print('done')
        self.logger.info("START FROM EPOCH {:d}, LOSS = {:.4f}".format(
            self.cur_epoch, best_loss))
        while self.cur_epoch < num_epochs:
            stats["title"] = "Loss(time/N, lr={:.3e}) - Epoch {:2d}:".format(
                self.optimizer.param_groups[0]["lr"],
                self.cur_epoch + 1)
            tr_loss, tr_batch, tr_cost = self.train(train_loader)
            print('End epoch {:d}, train loss: {:.2f}'.format(self.cur_epoch, tr_loss))
            stats["tr"] = "train = {:+.4f}({:.2f}s/{:d})".format(
                tr_loss, tr_cost, tr_batch)
            cv_loss, cv_batch, cv_cost = self.evaluate(dev_loader)
            stats["cv"] = "dev = {:+.4f}({:.2f}s/{:d})".format(
                cv_loss, cv_cost, cv_batch)
            stats["scheduler"] = ""
            if self.cur_epoch > warm_up_epochs:
                if cv_loss > best_loss:
                    stats["scheduler"] = "| no impr, best = {:.4f}".format(
                        self.scheduler.best)
                else:
                    best_loss = cv_loss
                    self.save_checkpoint(best=True)
                # schedule here
                self.scheduler.step(cv_loss)
            self.logger.info(
                "{title} {tr} | {cv} {scheduler}".format(**stats))
            # flush scheduler info
            sys.stdout.flush()
            # save checkpoint
            self.cur_epoch += 1
            self.save_checkpoint(best=False)

        self.logger.info(
            "Training for {:d} epoches done!".format(num_epochs))


class SiSnrTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(SiSnrTrainer, self).__init__(*args, **kwargs)

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

    def lfb_mse(self, x, s, eps=1e-8):
        """
        est_spec, ref_spec: BS x F x T
        return: log fbank MSE: BS tensor
        """
        if x.shape != s.shape:
            if x.shape[-1] > s.shape[-1]:
                x = x[:,:,:s.shape[-1]]
            else:
                s = s[:,:,:x.shape[-1]]

        lfb_x = self.nnet.lfb(x.permute((0,2,1)), self.nnet.epsilon)
        lfb_s = self.nnet.lfb(s.permute((0,2,1)), self.nnet.epsilon)
        t = torch.sum((lfb_x - lfb_s) ** 2, dim=-1)
        t = torch.sum(t, dim=-1)
        return t

    def sisnr(self, x, s, eps=1e-8):
        """
        Arguments:
        x: separated signal, BS x S
        s: reference signal, BS x S
        Return:
        sisnr: BS tensor
        """

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
        x_zm = x - torch.mean(x, dim=-1, keepdim=True)
        s_zm = s - torch.mean(s, dim=-1, keepdim=True)
        t = torch.sum(
            x_zm * s_zm, dim=-1,
            keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True) ** 2 + eps)
        return 20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))

    def compute_loss(self, data):
        '''
        compute SI-SNR loss with egs (samples)
        :param data: Tensor dict
        :return: loss: scalar
        '''
        # get results
        data = offload_data(data, self.device)

        # we need to divie ArkRunTools.C when using ark_scp
        with torch.no_grad():
            data["mix"] = data["mix"]/ArkRunTools.C
            data["ref"] = data["ref"]/ArkRunTools.C
            # self.logger.info("use_ark_scp=%s", use_ark_scp

        #ests, ests_noise, estspec = torch.nn.parallel.data_parallel(self.nnet, [data["mix"], data["src_doa"], data["spk_num"], data["lip_video"]],  device_ids=self.gpuid)
        ests, estspec  = torch.nn.parallel.data_parallel(self.nnet, [data["mix"], data["src_doa"], data["spk_num"], data["lip_video"], None],  device_ids=self.gpuid)
        #ests = self.nnet([data["mix"], data["src_doa"], data["spk_num"], data["lip_video"]])
        # spks x n x S
        num_spks = self.out_spk
        refs = data["ref"]
        refspec, refphase = self.nnet.df_computer.stft(refs)
        #import pdb; pdb.set_trace()
        def sisnr_loss(permute):
            # for one permute
            return sum([self.sisnr(ests[s], refs[:, t]) for s, t in enumerate(permute)]) / len(permute)

        def sisnr_loss_noise(permute):
            return sum([self.sisnr(ests_noise[s], (torch.squeeze(data["mix"][:,0,:]) - torch.squeeze(data["ref"])).unsqueeze(1)[:,t] ) for s, t in enumerate(permute)]) / len(permute)

        def lfb_loss(permute):
            return sum([self.lfb_mse(estspec[s], refspec) for s, t in enumerate(permute)]) / len(permute)
        
        #import pdb; pdb.set_trace()
        spec_mat = torch.stack([lfb_loss(p) for p in permutations(range(num_spks))])
        max_perutt_spec, _ = torch.max(spec_mat, dim=0)

        # P x N
        N = data["mix"].size(0)
        sisnr_mat = torch.stack(
            [sisnr_loss(p) for p in permutations(range(num_spks))])
        max_perutt, _ = torch.max(sisnr_mat, dim=0)
        loss_sisnr = -torch.sum(max_perutt) / N

        #sisnr_mat_noise = torch.stack([sisnr_loss_noise(p) for p in permutations(range(num_spks))])
        #max_perutt_noise, _ = torch.max(sisnr_mat_noise, dim=0)
        #loss_sisnr_noise = -torch.sum(max_perutt_noise) / N


        loss_lfb = torch.mean(max_perutt_spec) * 1e-4
        print(f"loss_sisnr : {loss_sisnr}, loss_lfb: {loss_lfb}")
        # si-snr
        return loss_sisnr, loss_lfb
