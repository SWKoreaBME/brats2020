from xml.etree.ElementInclude import include
import torch
import pandas as pd
import torch.nn as nn
import os

from monai.networks import one_hot
from monai.metrics import get_confusion_matrix


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


def compute_meandice_multilabel(seg_out, seg_label, include_background=False, sigmoid=True):
    """ Segmentation score for non-softmax output with multi-labels
    """
    num_labels = seg_out.size(1)
    sigmoid = nn.Sigmoid()
    if sigmoid:
        seg_out = torch.where(sigmoid(seg_out) > 0.5, 1, 0)
    
    seg_batch_score = 0
    for label_idx in range(num_labels):
        if (not include_background) and (label_idx == 0):
            continue
        _outs, _label = seg_out[:, label_idx], seg_label[:, label_idx]
        seg_batch_score = seg_batch_score + get_dice_score(_outs.detach().cpu(), _label.detach().cpu()).nanmean()
        if torch.isnan(seg_batch_score).any():
            print(seg_batch_score)
        
    if include_background:
        seg_batch_score /= num_labels
    else:
        seg_batch_score /= (num_labels - 1)
    return seg_batch_score


class Score:
    def __init__(self, include_background=False, sigmoid=False, softmax=False):
        self.total_dice = 0
        self.total_conf_matrix = None
        self.sigmoid_fn = nn.Sigmoid()
        self.sigmoid = sigmoid
        self.include_background = include_background
        self.total_num_imgs = 0
        self.total_data = list()
        self.tumor_names = ["Tumor Core (TC)", "Whole Tumor (WT)", "Enhancing Tumor (ET)"]
    
    @staticmethod
    def get_sen_spe(conf_matrix):
        tp, fp, tn, fn = conf_matrix
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        return sensitivity, specificity
        
    def save(self, ckpt_save_dir):
        dice_score = self.total_dice / self.total_num_imgs
        conf_matrix_sum = self.total_conf_matrix.sum(0)
        
        mean_sen, mean_spe = 0, 0
        for label_idx in range(conf_matrix_sum.size(0)):
            sen, spe = self.get_sen_spe(conf_matrix_sum[label_idx])
            mean_sen += sen
            mean_spe += spe
            
        mean_sen = mean_sen / len(self.tumor_names)
        mean_spe = mean_spe / len(self.tumor_names)
        
        print(f"Dice: {dice_score:.4f}")
        print(f"Sensitivity: {mean_sen:.4f}")
        print(f"Specificity: {mean_spe:.4f}")
        
        self.total_data.append([float(dice_score), float(mean_sen), float(mean_spe)])
        pd.DataFrame(self.total_data, 
                     columns=["Dice", "Sensitivity", "Specificity"]).to_csv(os.path.join(ckpt_save_dir, "test.csv"))
    
    def evaluate(self, y_pred, y_true):
        if self.sigmoid:
            y_pred = torch.where(self.sigmoid_fn(y_pred) > 0.5, 1, 0)
        
        y_pred = y_pred.detach().cpu()
        y_true = y_true.detach().cpu()
        dice_score = compute_meandice_multilabel(y_pred,
                                                 y_true, 
                                                 include_background=self.include_background, 
                                                 sigmoid=False) * y_pred.size(0)
        
        conf_matrix = get_confusion_matrix(y_pred,
                                           y_true,
                                           include_background=self.include_background).detach().cpu()
        
        if self.total_conf_matrix is None:
            self.total_conf_matrix = conf_matrix
        else:
            self.total_conf_matrix = torch.cat((conf_matrix, self.total_conf_matrix), 0)
        
        self.total_dice += dice_score
        self.total_num_imgs += y_pred.size(0)
    

if __name__ == "__main__":
    pass
