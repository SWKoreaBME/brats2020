import torch
import pandas as pd


def concat_bg(tensor):
    """Make One-hot for already one-hot encoded tensor, just in case the encoded tensor does not include background
    
        - tensor, torch.Tensor, B x C x ...
    """
    
    batch_size = tensor.size(0)
    tensor_size = tensor.size()[2:]
    
    bg_tensor = torch.zeros((batch_size, *tensor_size))
    bg_tensor[torch.where(tensor.sum(1) == 0)] = 1
    out_tensor = torch.cat([bg_tensor.unsqueeze(1), tensor], 1).to(tensor.device)
    return out_tensor


def count_parameters(model):
    """Calculate the number of paramters of a given model
    Code from viswanatha_reddy
    https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/24
    """
    table = list()
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.append([name, param])
        total_params+=param
    table_df = pd.DataFrame(data=table, columns=["Modules", "Parameters"])
    print(f"Total Trainable Params: {total_params}")
    return table_df, total_params


if __name__== "__main__":
    pass
