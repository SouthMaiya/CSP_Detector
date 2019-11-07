import torch
import torch.nn as nn
import numpy as np
import ipdb


def get_mask(IMAGE_WIDTH, IMAGE_HEIGHT, center_x, center_y):
    # center_x = IMAGE_WIDTH / 2
    # center_y = IMAGE_HEIGHT / 2
    R = np.sqrt(center_x ** 2 + center_y ** 2)

    mask_x = (torch.ones(size=(IMAGE_HEIGHT, IMAGE_WIDTH)) * center_x).cuda()
    mask_y = (torch.ones(size=(IMAGE_HEIGHT, IMAGE_WIDTH)) * center_y).cuda()

    x1 = (torch.arange(IMAGE_WIDTH)).cuda()
    x_map = (x1.repeat((IMAGE_HEIGHT, 1)).float()).cuda()

    y1 = (torch.arange(IMAGE_HEIGHT)).cuda()
    y_map = (y1.repeat((IMAGE_WIDTH, 1)).permute(1, 0).float()).cuda()

    Gauss_map = (torch.sqrt((x_map - mask_x) ** 2 + (y_map - mask_y) ** 2)).cuda()
    Gauss_map = (torch.exp(-0.5 * Gauss_map / R)).cuda()
    return Gauss_map



class loss(nn.Module):
    def __init__(self,alpha=1.0,gamma=2.0,beta=4.0):
        super(loss,self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta



    def forward(self,center_maps,scale_maps,annotations,stride=4):
        batch_size = center_maps.size()[0]
        scale_losses = []
        center_losses = []
        for i in range(batch_size):
            boxes = annotations[i]
            center_map = center_maps[i]
            scale_map = scale_maps[i]
            boxes = (boxes//stride).long()
            center_map = torch.clamp(center_map, 1e-4, 1.0 - 1e-4)
            x1,y1,x2,y2 = boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
            center_x,center_y,width,height = (x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1
            center_gt = torch.zeros(center_map.shape).cuda()
            #
            #print(center_gt.size())
            scale_gt = torch.zeros(scale_map.shape).cuda()
            center_gt[:,center_y,center_x] = 1.0
            region_x = torch.cat([center_x-2,center_x-1,center_x,center_x+1,center_x+2])
            region_y = torch.cat([center_y-2,center_y-1,center_y,center_y+1,center_y+2])
            scale_gt[:,region_y.cuda(),region_x.cuda()] = (torch.log(height.float())).repeat(5,).cuda()
            Gauss_map = torch.zeros(center_map.shape).cuda()
            pos_map = torch.zeros(center_map.shape).cuda()
            K = boxes.size()[0]

            for i in range(K):
                c_x,c_y,w,h=center_x[i],center_y[i],width[i],height[i]
                k_Gauss= get_mask(w,h,c_x,c_y)
                Gauss_map[:,y1[i]:y2[i],x1[i]:x2[i]] = torch.max(k_Gauss.unsqueeze(0),
                            Gauss_map[:,y1[i]:y2[i],x1[i]:x2[i]])
                pos_map[:,y1[i]:y2[i],x1[i]:x2[i]] = 1

            Gauss_map = torch.pow(1.0-Gauss_map,self.beta)
            Gauss_map = Gauss_map * pos_map
            #ipdb.set_trace()


            alpha_factor = torch.ones(center_map.shape).cuda() * self.alpha
            alpha_factor = torch.where(torch.eq(center_gt, 1.), alpha_factor, Gauss_map)
            focal_weight = torch.where(torch.eq(center_gt, 1.),1.0-center_map ,center_map)
            focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)
            bce = -(center_gt * torch.log(center_map) + (1.0 - center_gt) * torch.log(1.0 - center_map))
            center_loss = focal_weight * bce
            center_loss = center_loss.sum()/max(1.0,K)

            center_losses.append(center_loss)


            scale_diff = torch.abs(scale_gt - scale_map)
            scale_loss = torch.where(
                torch.le(scale_diff, 1.0  ),
                0.5  * torch.pow(scale_diff, 2),
                scale_diff - 0.5
            )

            scale_loss = torch.where(torch.ne(scale_gt,0.), scale_loss, torch.zeros(scale_loss.shape).cuda())

            scale_losses.append(scale_loss.sum()/max(1.0,K))

            return torch.stack(center_losses).mean(dim=0, keepdim=True), torch.stack(scale_losses).mean(
                dim=0, keepdim=True)
