import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_with_hist(imgs: list, img_titles: list, img_save_path=None, show: bool = True) -> None:
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
        cmap = 'Spectral_r' if 'uncertainty' in img_titles[img_idx].lower() else 'gray'
        
        axes[f'ax_{img_idx + 1}_image'] = fig.add_subplot(2, num_imgs, img_idx + 1)
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
    
    
if __name__ == "__main__":
    pass

