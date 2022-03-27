from typing import Sequence, Tuple, Union

from monai.networks.nets import UNETR
from monai.networks.nets.vit import ViT
from easydict import EasyDict as edict


class UNETR(UNETR):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    
    Code changed by Sangwook
    """
    
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
        self.num_layers = 4  # for training Brats 2020 model
        args = edict(kwargs)
        
        self.vit = ViT(
            in_channels=args.in_channels,
            img_size=args.img_size,
            patch_size=self.patch_size,
            hidden_size=args.hidden_size,
            mlp_dim=args.mlp_dim,
            num_layers=self.num_layers,
            num_heads=args.num_heads,
            pos_embed=args.pos_embed,
            classification=self.classification,
            dropout_rate=args.dropout_rate,
            spatial_dims=args.spatial_dims,
        )

    def forward(self, x_in):
        """Minor changes for different number of layers in UNETR
        """
        
        if self.num_layers == 12:
            idx1, idx2, idx3 = 3, 6, 9
            
        elif self.num_layers == 4:
            idx1, idx2, idx3 = 1, 2, 3
            
        x, hidden_states_out = self.vit(x_in)
        enc1 = self.encoder1(x_in)
        x2 = hidden_states_out[idx1]
        enc2 = self.encoder2(self.proj_feat(x2, self.hidden_size, self.feat_size))
        x3 = hidden_states_out[idx2]
        enc3 = self.encoder3(self.proj_feat(x3, self.hidden_size, self.feat_size))
        x4 = hidden_states_out[idx3]
        enc4 = self.encoder4(self.proj_feat(x4, self.hidden_size, self.feat_size))
        dec4 = self.proj_feat(x, self.hidden_size, self.feat_size)
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)
        return self.out(out)
    

if __name__=="__main__":
    import torch
    
    device = "cpu"
    batch_size = 2
    
    # UNETR
    model_params = dict(
        in_channels = 3,
        out_channels = 1,
        feature_size = 8,
        img_size = (512, 384),
        hidden_size = 768,
        mlp_dim = 128,  # hyper-params (hp) 1
        num_heads = 8,  # hyper-params (hp) 2
        pos_embed = "conv",
        dropout_rate = 0.3,
        spatial_dims = 2,
        norm_name = "instance",
        conv_block = True,
        res_block = True
    )
    
    model = UNETR(model_params).to(device)
    x = torch.randn((batch_size, 3, 512, 384)).to(device)
    seg_out = model(x)
    
    pass