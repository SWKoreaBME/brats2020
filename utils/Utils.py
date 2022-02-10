import torch


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


if __name__== "__main__":
    pass
