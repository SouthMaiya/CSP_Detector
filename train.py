import torch
from utils.config import Config
from models.CSPDetector import CSPDetector
from data.build import build_dataloader
from utils.util import write_

def main():
    cfg = Config()
    model =CSPDetector(cfg)
    train_dataset = build_dataloader(cfg,'train')
    val_dataset = build_dataloader(cfg,'test')
    best_ap = 0
    for epoch in range(cfg.enpoch):
        for iter_num,data in enumerate(train_dataset):
            img,annotations = data['img'].cuda().float(), data['annot']
            model.set_input(img,annotations)
            center_loss,scale_loss = model.optimize()
            print('Epoch: {} | Iteration: {} | center loss: {:1.5f} | scale loss: {:1.5f}'.format(epoch,
                                        iter_num, float(center_loss), float(scale_loss)))
        if epoch % cfg.eval_poch == 0:
            print('start eval×*×**×*×*×*×*×')
            recall,precision,ap = model.eval(val_dataset)
            if ap > best_ap:
                best_ap = ap
                message = "epoch：{}\nrecall：{:.4f},ap：{:.4f}".format(epoch,recall,ap)
                write_(cfg.export_dir,cfg.dataname,message)



if __name__ == '__main__':
    main()


