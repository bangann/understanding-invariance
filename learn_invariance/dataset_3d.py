from PIL import Image
import numpy as np
import os
import os.path
from typing import Any, Callable, List, Optional, Tuple
import json
from torch.utils.data import Dataset


def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename: str) -> bool:
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path: str) -> Image.Image:
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


# TODO: specify the return type
def accimage_loader(path: str) -> Any:
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class R2N2RenderingsTorch(Dataset):
    """
    load metainfo containing splits of model ids
    load {cat_id}/{model_id}/rendering/00.png
    randomly load 2 extra pngs from {01.png, ..., 23.png}

    """

    def __init__(
            self,
            root: str,
            metainfo_path: str = None,
            transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            train: bool = True,
            download=None,
    ):
        self.loader = loader
        self.transform = transform

        self.root = root

        split = 'train' if train else 'test'
        self.samples = self._load_metainfo(split, metainfo_path)

    def _load_metainfo(self, split, path):
        """
        metainfo: {'train': (
        split: 'train', 'test'

        Returns:
            List[Tuple[str, int]]: samples of a form (path_to_sample, class)
        """

        if path == None:
            path = os.path.join(self.root, '..', 'metainfo.json')
        with open(path) as f:
            d = json.load(f)

        return d[split]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]

        views = ['00',
                 '{:02}'.format(np.random.randint(0, 24)),
                 '{:02}'.format(np.random.randint(0, 24))]

        # uncomment this when evaluating 'reg' method in order to get the worst case agreement(consistency accuracy)
        # views = ['00',
        #          '01',
        #          '02',
        #          '03',
        #          '04',
        #          '05',
        #          '06',
        #          '07',
        #          '08',
        #          '09',
        #          '10',
        #          '11',
        #          '12',
        #          '13',
        #          '14',
        #          '15',
        #          '16',
        #          '17',
        #          '18',
        #          '19',
        #          '20',
        #          '21',
        #          '22',
        #          '23']

        samples, targets = [], []
        for view in views:
            path_abs = os.path.join(self.root, path, 'rendering', view + '.png')
            sample = self.loader(path_abs)
            if self.transform is not None:
                sample = self.transform(sample)

            samples.append(sample)
            targets.append(target)

        return samples, targets

    def __len__(self) -> int:
        return len(self.samples)
