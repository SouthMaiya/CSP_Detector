from .HRSC import HRSCDataset
from .transform import *
from torchvision import  transforms

def build_dataloader(cfg,datatype):
    if datatype == 'train':
        data_dir = cfg.datadir+'/'+'Train'
        dataset_train = HRSCDataset(data_dir=data_dir,transform=transforms.Compose(
            [Normalizer(), Augmenter(), Resizer()]))
        sampler = AspectRatioBasedSampler(dataset_train, batch_size=cfg.batch_size, drop_last=False)
        dataset= DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)
    if datatype == 'test':
        data_dir = cfg.datadir + '/' + 'Test'
        dataset= HRSCDataset(data_dir=data_dir,transform=transforms.Compose([Normalizer(), Resizer()]))
        # sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        #
        # dataloader = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)


    return dataset