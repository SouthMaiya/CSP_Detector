import torch
import torch.nn as nn
from .build_net import Build_Net
from .loss import loss
from utils.util import *
from utils.eval import model_eval

class CSPDetector(object):
    def __init__(self,opt):
        self.net = Build_Net(opt.model).cuda()
        self.loss = loss()
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=opt.lr)


    def set_input(self, img_batch, annotations, img_id=None):
        self.img = img_batch
        self.ann = annotations
        self.img_id = img_id


    def optimize(self):
        self.optimizer.zero_grad()
        center_map,scale_map = self.net.forward(self.img)
        center_loss,scale_loss = self.loss(center_map,scale_map,self.ann)
        loss = center_loss + scale_loss
        loss.backward()
        self.optimizer.step()
        return center_loss,scale_loss

    def eval(self,dataset_val):
        return model_eval(dataset_val,self.net)


    def save(self, path):
        mkdir(path)
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path))


    def update_lr(self, new_lr, factor=False):
        if factor:
            new_lr = self.old_lr / new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

