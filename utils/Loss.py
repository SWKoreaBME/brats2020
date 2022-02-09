import torch
import torch.nn as nn

from monai.networks import one_hot

class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1e-4, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def get_dice_score(self, predict: torch.Tensor, target: torch.Tensor, smooth=1e-4, p=2):
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = 2 * torch.sum(torch.mul(predict, target), dim=1) + smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + smooth

        return num / den

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"

        dice_score = self.get_dice_score(predict, target)

        loss = 1 - dice_score

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))
        
        
def get_dice_score(self, predict: torch.Tensor, target: torch.Tensor, smooth=1e-4, p=2):
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)

    num = 2 * torch.sum(torch.mul(predict, target), dim=1) + smooth
    den = torch.sum(predict.pow(self.p) + target.pow(p), dim=1) + smooth

    return num / den


def seg_loss_fn_3d(seg_out:torch.Tensor,
                   seg_label:torch.Tensor, 
                   lambda_ce:float = 0.7,
                   lambda_dice:float = 0.3,
                   input_is_softmax:bool = False, 
                   include_background:bool = False):
    """Calculate segmentation loss for 3D images
    """
    
    device = seg_out.device
    ce_loss_function = nn.CrossEntropyLoss().to(device)
    dice_loss_function = BinaryDiceLoss(smooth=1e-4, p=2, reduction='mean').to(device)
    softmax = nn.Softmax(1)
    
    # ce loss
    B, num_labels = seg_out.size(0), seg_out.size(1)
    if not input_is_softmax:
        seg_out = softmax(seg_out)
    
    if lambda_ce == 0:
        tmp_ce_loss = 0
    else:
        tmp_ce_loss = ce_loss_function(seg_out.view(B, num_labels, -1), torch.argmax(seg_label, 1).long().view(B, -1))
        if torch.isnan(tmp_ce_loss).any():
            print(tmp_ce_loss)
            pass
        
    # make binary output (one-hot)
    seg_out = one_hot(seg_out.argmax(1).unsqueeze(1), num_classes=4)  # make one-hot

    # dice loss
    if lambda_dice == 0:
        tmp_dice_loss = 0
    else:
        tmp_dice_loss = 0
        for label_idx in range(num_labels):
            if (not include_background) and (label_idx==0):
                continue
            _outs = seg_out[:, label_idx]
            _label = seg_label[:, label_idx]
            dice_loss = dice_loss_function(_outs, _label)
            tmp_dice_loss = tmp_dice_loss + dice_loss
            
            if torch.isnan(dice_loss).any():
                print(dice_loss)
                pass

        if include_background:
            tmp_dice_loss /= num_labels
        else:
            tmp_dice_loss /= (num_labels - 1)
        
    seg_loss_value = tmp_ce_loss * lambda_ce + tmp_dice_loss * lambda_dice
    return seg_loss_value / num_labels


if __name__ == "__main__":
    pass
