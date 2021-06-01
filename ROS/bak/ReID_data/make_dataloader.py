from PIL import Image
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .DTC2021 import DTC2021
from .bases import ImageDataset




def val_collate_fn(batch):
    """ related to 'ImageDataset' class 
        - (ref) https://github.com/DoranLyong/person-reid-tiny-baseline/blob/master/datasets/bases.py
    """
    imgs, pids, img_paths = zip(*batch)
    return torch.stack(imgs, dim=0), pids, img_paths


def make_dataloader(cfg):

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    num_workers = cfg.DATALOADER_NUM_WORKERS
    dataset = DTC2021(data_dir=cfg.DATA_DIR, verbose=True)

    num_classes = dataset.num_gallery_pids



    gallery_set = ImageDataset(dataset.gallery, val_transforms)
    gallery_loader = DataLoader(gallery_set,
                            batch_size=cfg.TEST_IMS_PER_BATCH,
                            shuffle=False, num_workers=num_workers,
                            collate_fn=val_collate_fn
                            )
    return gallery_loader, num_classes
