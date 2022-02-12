import torch
import torch.nn as nn
from collections import OrderedDict


def load_model_weights(model, weight_path, dp=True, device="cuda"):
    if device == "cuda":
        if dp:
            model = nn.DataParallel(model)
            model.load_state_dict(torch.load(weight_path))
        else:
            state_dict = torch.load(weight_path)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if 'submodule' in k:
                    new_state_dict[k] = v
                    continue
                else:
                    name = k.replace('module.', '')  # remove `module.`
                    new_state_dict[name] = v
            model.load_state_dict(new_state_dict)

    elif device=="cpu":
        state_dict = torch.load(weight_path, map_location=torch.device('cpu'))
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '')  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        
    print(f"Model Loaded from {weight_path}, Multi GPU: {dp}")
    return model


if __name__ == "__main__":
    pass
