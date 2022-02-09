import os
import numpy as np
from glob import glob

from monai.data import DataLoader, Dataset


def load_dataloader(root_dir, phase, transforms, dataloader_params):
    """Return dataloader
    
        -------
        root_dir = "/cluster/home/kimsa/data/BraTs2020/BraTS2020_training_data/content/data_monai"
    
        -------
        Examples
        
        for batch_idx, batch in enumerate(train_dataloader):
            image, label = batch["image"], batch["label"]
    """
    num_train = 260  # 70%, 40300 slices (volume 1 ~ 260)
    num_valid = 40 # 10% 6200 slices (volume 261 ~ 300)
    num_test = 69 # 20% 10695 slices (volume 301 ~ 369)
    
    image_list = sorted(glob(os.path.join(root_dir, "image", "*.nii.gz")))
    label_list = sorted(glob(os.path.join(root_dir, "label", "*.nii.gz")))
    
    get_volume_name = lambda x: os.path.basename(x).split("_slice")[0]
    sort_key = lambda x: int(x.split('_')[-1])
    unique_volumes = sorted(np.unique([get_volume_name(x) for x in image_list]), key=sort_key)

    train_subjects, valid_subjects, test_subjects = unique_volumes[:num_train], unique_volumes[num_train:num_train+num_valid], unique_volumes[num_train+num_valid:]
    
    if phase == "train":
        train_imgs = [x for x in image_list if get_volume_name(x) in train_subjects]
        train_lbls = [x for x in label_list if get_volume_name(x) in train_subjects]
        train_list = [
            {'image': image_name, 'label': label_name}
            for image_name, label_name in zip(train_imgs, train_lbls)
        ]
        file_list = train_list
        
    elif phase == "valid":
        valid_imgs = [x for x in image_list if get_volume_name(x) in valid_subjects]
        valid_lbls = [x for x in label_list if get_volume_name(x) in valid_subjects]
        valid_list = [
            {'image': image_name, 'label': label_name}
            for image_name, label_name in zip(valid_imgs, valid_lbls)
        ]
        file_list = valid_list
        
    elif phase == "test":
        test_imgs = [x for x in image_list if get_volume_name(x) in test_subjects]
        test_lbls = [x for x in label_list if get_volume_name(x) in test_subjects]
        test_list = [
            {'image': image_name, 'label': label_name}
            for image_name, label_name in zip(test_imgs, test_lbls)
        ]
        file_list = test_list

    dataset = Dataset(data=file_list, transform=transforms)
    dataloader = DataLoader(dataset, **dataloader_params)

    return dataloader


if __name__ == "__main__":
    # Example

    import monai.transforms as tf
    from monai.networks import one_hot

    # train transforms
    train_transforms = tf.Compose([
        tf.LoadImaged(reader="NibabelReader", keys=['image', 'label']),
        # tf.AsDiscreted(keys=['label'], threshold_values=True),
        tf.ToNumpyd(keys=['image', 'label']),
        tf.NormalizeIntensityd(keys=['image'], channel_wise=True, nonzero=True),
        tf.ToTensord(keys=['image', 'label']),
    ])

    # validation and test transforms
    val_transforms = tf.Compose([
        tf.LoadImaged(reader="NibabelReader", keys=['image', 'label']),
        # tf.AsDiscreted(keys=['label'], threshold_values=True),
        tf.ToNumpyd(keys=['image', 'label']),
        tf.NormalizeIntensityd(keys=['image'], channel_wise=True, nonzero=True),
        tf.ToTensord(keys=['image', 'label'])
    ])

    root_dir = "/cluster/projects/mcintoshgroup/BraTs2020/data_monai/"

    loader_params = dict(
        batch_size=8,
        shuffle=True
    )

    train_dataloader = load_dataloader(root_dir, "train", train_transforms, loader_params)
    valid_dataloader = load_dataloader(root_dir, "valid", val_transforms, loader_params)
    test_dataloader = load_dataloader(root_dir, "test", val_transforms, loader_params)

    num_epochs = 2
    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(train_dataloader):
            inputs, labels = batch["image"], batch["label"]
            labels = one_hot(labels.argmax(1).unsqueeze(1), num_classes=4)  # make one-hot

            print(f"Input Size: {inputs.size()}")
            print(f"Input Size: {labels.size()}")

            break
        break
