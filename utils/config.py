class Config():
    batch_size = 1
    lr = 0.000001
    #datadir ="E:/data/object_dat/HSRC2016/HRSC2016part2/HRSC2016"
    datadir = '/media/zxq/data/data/object_data/HSRC2016/HRSC2016_part01'
    model = 'resnet50'
    enpoch = 200
    eval_poch = 5
    checkpoints_dir ='./checkpoints'
    dataname = 'HRSC'
    export_dir = './result'

