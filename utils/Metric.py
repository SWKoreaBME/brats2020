import torch
import torch.nn as nn
from monai.networks import one_hot


def get_dice_score(predict: torch.Tensor, target: torch.Tensor, smooth=1e-4, p=2):
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)

    num = 2 * torch.sum(torch.mul(predict, target), dim=1) + smooth
    den = torch.sum(predict.pow(p) + target.pow(p), dim=1) + smooth

    return num / den


def compute_meandice(seg_out, seg_label, include_background=False):
    """ Segmentation score for non-softmax output with multi-labels
    """
    num_labels = seg_out.size(1)
    softmax = nn.Softmax(1)
    seg_out = one_hot(softmax(seg_out).argmax(1, keepdim=True), num_classes=num_labels)  # make one-hot
    seg_label = one_hot(seg_label.argmax(1, keepdim=True), num_classes=num_labels)  # make one-hot
    
    seg_batch_score = 0
    for label_idx in range(num_labels):
        if (not include_background) and (label_idx == 0):
            continue
        _outs, _label = seg_out[:, label_idx], seg_label[:, label_idx]
        seg_batch_score = seg_batch_score + get_dice_score(_outs.detach().cpu(), _label.detach().cpu()).nanmean()
        if torch.isnan(seg_batch_score).any():
            print(seg_batch_score)
            pass
        pass
        
    if include_background:
        seg_batch_score /= num_labels
    else:
        seg_batch_score /= (num_labels - 1)
    return seg_batch_score

if __name__ == "__main__":
    pass
