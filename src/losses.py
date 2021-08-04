import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms



def tv_loss(x, beta = 0.5, reg_coeff = 5):
    '''Calculates TV loss for an image `x`.
        
    Args:
        x: image, torch.Variable of torch.Tensor
        beta: See https://arxiv.org/abs/1412.0035 (fig. 2) to see effect of `beta` 
    '''
    dh = torch.pow(x[:,:,:,1:] - x[:,:,:,:-1], 2)
    dw = torch.pow(x[:,:,1:,:] - x[:,:,:-1,:], 2)
    a,b,c,d=x.shape
    return reg_coeff*(torch.sum(torch.pow(dh[:, :, :-1] + dw[:, :, :, :-1], beta))/(a*b*c*d))


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss


class LossWithMask(nn.Module):

    def __init__(self, loss_type='mse', eps=1e-3):
        super(LossWithMask, self).__init__()
        self.loss_type = loss_type
        self.eps = eps

    def maskedMSELoss(self, pred, target, mask):
        diff2 = (torch.flatten(pred) - torch.flatten(target)) ** 2.0 * torch.flatten(mask)
        mse = torch.sum(diff2) / (torch.sum(mask) + self.eps)
        return mse

    def forward(self, pred, target, input):
        # mask if all channels are great and equal 0.98 (flare area)
        mask = torch.zeros(input.shape[0], 1, input.shape[2], input.shape[3]).cuda()
        for i in range(input.shape[0]):
            mask[i,0,:,:] = torch.logical_and(input[i,0,:,:].ge(0.99), input[i,1,:,:].ge(0.93))
            mask[i,0,:,:] = ~torch.logical_and(mask[i,0,:,:], input[i,2:,:,:].ge(0.99))
            mask = mask.expand(-1, 3, -1, -1)
        return self.maskedMSELoss(pred, target, mask)


class ContentLossWithMask(nn.Module):
    
    def __init__(self, loss_type='mse', eps=1e-3):
        super(ContentLossWithMask, self).__init__()
        if loss_type == 'mse':
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.L1Loss()
        self.eps = eps
        
        # vgg-19
        with torch.no_grad():
            self.model = self.vgg19partial()
            self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def vgg19partial(self):
        conv_3_3_layer = 14
        cnn = models.vgg19(pretrained=True).features
        cnn = cnn.cuda()
        model = nn.Sequential()
        model = model.cuda()
        model = model.eval()
        for i, layer in enumerate(list(cnn)):
            model.add_module(str(i), layer)
            if i == conv_3_3_layer:
                break
        return model

    def forward(self, pred, target, input):
        # mask(set to 0) if all channels are great and equal 0.98 (flare area)
        mask = torch.zeros(input.shape[0], 1, input.shape[2], input.shape[3]).cuda()
        for i in range(input.shape[0]):
            mask[i,0,:,:] = torch.logical_and(input[i,0,:,:].ge(0.99), input[i,1,:,:].ge(0.93))
            mask[i,0,:,:] = ~torch.logical_and(mask[i,0,:,:], input[i,2:,:,:].ge(0.99))
            mask = mask.expand(-1, 3, -1, -1)

        # Copy target masked image into pred one
        pred_c = target * (1-mask) + pred * (mask)

        # vgg19 inputs(pred/target) should be normalized as [0, 1]
        t_pred_c = pred_c.clone()
        t_target = target.clone()
        for i in range(input.shape[0]):
            t_pred_c[i,:,:,:] = self.transform(t_pred_c[i,:,:,:])
            t_target[i,:,:,:] = self.transform(t_target[i,:,:,:])
        f_pred_c = self.model.forward(t_pred_c)
        f_target = self.model.forward(t_target)

        # MSE loss + Content loss
        loss_m = self.criterion(pred_c, target)
        loss_c = self.criterion(f_pred_c, f_target.detach())
        return 0.5 * loss_m + 0.006 * loss_c

class bootstrapped_cross_entropy2d(torch.nn.modules.loss._Loss):
    def __init__(self, size_average=True, reduce=False, reduction='mean'):
        super(bootstrapped_cross_entropy2d, self).__init__(size_average, reduce, reduction)
        self.size_average = size_average
        self.reduce = False
        
    def forward(self, input, target, min_K=4096, loss_th=0.3, weight=None):
        n, c, h, w = input.size()
        nt, ht, wt = target.size()
        batch_size = input.size()[0]

        if h != ht and w != wt:  # upsample labels
            input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

        thresh = loss_th

        def _bootstrap_xentropy_single(input, target, K, thresh, weight=None, size_average=True):

            n, c, h, w = input.size()
            input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
            target = target.view(-1)

            loss = F.cross_entropy(
                input, target, weight=weight, reduce=self.reduce, size_average=self.size_average, ignore_index=250
            )
            sorted_loss, _ = torch.sort(loss, descending=True)

            if sorted_loss[K] > thresh:
                loss = sorted_loss[sorted_loss > thresh]
            else:
                loss = sorted_loss[:K]
            reduced_topk_loss = torch.mean(loss)

            return reduced_topk_loss

        loss = 0.0
        # Bootstrap from each image not entire batch
        for i in range(batch_size):
            loss += _bootstrap_xentropy_single(
                input=torch.unsqueeze(input[i], 0),
                target=torch.unsqueeze(target[i], 0),
                K=min_K,
                thresh=thresh,
                weight=weight,
                size_average=self.size_average,
            )
        return loss / float(batch_size)
