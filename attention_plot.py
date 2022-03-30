import os
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_with_hist(imgs: list, img_titles: list, img_save_path=None, show: bool = True, cmaps=None) -> None:
    """Plot given images with corresponding histograms
    """
    num_imgs = len(imgs)
    fig = plt.figure(figsize=(40, 4 * num_imgs))
    axes = dict()
    for img_idx in range(num_imgs):
        img = imgs[img_idx]
        if np.isnan(img).all():
            img = np.zeros_like(img)
        if len(img.shape) == 3:
            img = img[img.shape[0] // 2]  # if the image is 3d, show the center slice
        axes[f'ax_{img_idx + 1}_image'] = fig.add_subplot(2, num_imgs, img_idx + 1)
        cmap = 'gray' if cmaps is None else cmaps[img_idx]
        im1 = axes[f'ax_{img_idx + 1}_image'].imshow(img, cmap=cmap)
        divider = make_axes_locatable(axes[f'ax_{img_idx + 1}_image'])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im1, cax=cax, orientation='vertical')
        axes[f'ax_{img_idx + 1}_hist'] = fig.add_subplot(2, num_imgs, img_idx + num_imgs + 1)
        im2 = axes[f'ax_{img_idx + 1}_hist'].hist(img.flatten(), bins=50)
        axes[f'ax_{img_idx + 1}_image'].set_title(f"{img_titles[img_idx]}", fontsize=20)
        axes[f'ax_{img_idx + 1}_image'].axis('off')
        axes[f'ax_{img_idx + 1}_hist'].set_title(f"{img_titles[img_idx]} Histogram", fontsize=20)
        
    if img_save_path is not None:
        plt.savefig(img_save_path, bbox_inches='tight', dpi=50)
        
    if show:
        plt.show()
    plt.close()


attention_map_file = "./asset/attention_plot.pkl"
with open(attention_map_file, 'rb') as f:
    data = pkl.load(f)
    
    for b in range(5):
        img_np = data["img"][b, :, :, 0]
        attn_np_1 = data["attn_1"][b]
        attn_np_2 = data["attn_1"][b]
        
        img_save_path = os.path.join("asset", f"attention_map-{b}.jpg")
        
        fig, axes = plt.subplots(2, 2, figsize=(35, 30))
        
        ax1 = axes[0, 0]
        ax1.imshow(img_np, cmap='gray')
        ax1.axis('off')
        
        ax2 = axes[1, 0]
        ax2.imshow(img_np, cmap='gray')
        ax2.imshow(attn_np_1, cmap='Spectral', alpha=0.3)
        ax2.axis('off')
        
        ax3 = axes[0, 1]
        ax3.imshow(img_np, cmap='gray')
        ax3.axis('off')
        
        ax4 = axes[1, 1]
        ax4.imshow(img_np, cmap='gray')
        ax4.imshow(attn_np_2, cmap='Spectral', alpha=0.3)
        ax4.axis('off')
        
        plt.savefig(img_save_path, bbox_inches='tight', dpi=100)