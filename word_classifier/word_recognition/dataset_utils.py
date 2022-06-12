from typing import Optional, Callable, Union
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import torchtext as tt


class TRDGDataset(Dataset):
    """
    Dataset generated with `trdg` library (https://pypi.org/project/trdg/).
    A collection of `[word name]_[index].jpg` images in the `root` directory.
    """
    def __init__(self, root: str, transform: Optional[Callable] = None,
                 img_format: str = 'jpg'):
        assert os.path.exists(root), f'Invalid dataset path {root}'
        self.files = tuple(os.path.join(root, f) for f in os.listdir(root)
                           if f.endswith(img_format))
        self.transform = transform
        self.to_tensor = T.ToTensor()

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str, int]:
        file = self.files[index]
        label = self._get_label(file)
        with Image.open(file, 'r') as img:
            x = self.transform(img) if self.transform is not None else img
            if not isinstance(x, torch.Tensor):
                x = self.to_tensor(x)
        return x, label, len(label)

    @staticmethod
    def _get_label(file: str) -> str:
        fname = os.path.split(file)[1]
        return fname.split('_')[0]


class WordDataset(Dataset):
    def __init__(self, root: str, transform: Optional[Callable] = None,
                 img_format: str = 'jpg'):
        # check if dataset exists in `root`
        self.label_file = os.path.join(root, 'labels.txt')
        self.img_dir = os.path.join(root, 'images')
        assert all(os.path.exists(p)
                   for p in (root, self.label_file, self.img_dir))

        # find all images
        img_files = []
        for dirpath, _, filenames in os.walk(self.img_dir):
            for filename in filenames:
                if filename.endswith(img_format):
                    img_files.append(os.path.join(dirpath, filename))
        img_files.sort(key=lambda f: int(os.path.split(f)[-1].split('.')[0]))
        self.img_files = tuple(img_files)

        # read image labels (words)
        labels = []
        with open(self.label_file, 'r') as infile:
            for line in infile:
                word = line.rstrip()
                if word:  # skip blank lines
                    labels.append(word)
        self.labels = tuple(labels)
        assert len(self.labels) == len(self.img_files)

        # save transforms
        self.transform = transform
        self.to_tensor = T.ToTensor()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str, int]:
        with Image.open(self.img_files[index], 'r') as img:
            x = self.transform(img) if self.transform is not None else img
            if not isinstance(x, torch.Tensor):
                x = self.to_tensor(x)
        label = self.labels[index]
        return x, label, len(label)


def make_char_vocab(dataset: Dataset) -> tt.vocab.Vocab:
    # very slow, do not use with large datasets
    chars = set()
    for _, label, _ in dataset:
        for ch in label:
            chars.add(ch)
    return tt.vocab.build_vocab_from_iterator(
        chars, specials=['<->'], special_first=True)


def collate_fn(batch: list[tuple], vocab: Union[tt.vocab.Vocab, dict],
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    imgs, labels, target_lengths = zip(*batch)

    # pad images with zeros
    nc, h, w = map(max, zip(*[img.size() for img in imgs]))
    new_imgs = []
    for img in imgs:
        new_img = torch.zeros(nc, h, w)
        _, h_i, w_i = img.size()
        new_img[:, :h_i, :w_i] = img
        new_imgs.append(new_img)

    # convert labels to tensors of tokens
    targets = torch.zeros(sum(target_lengths), dtype=torch.int64)
    i1 = 0
    for label, label_len in zip(labels, target_lengths):
        targets[i1:i1+label_len] = torch.tensor([vocab[ch] for ch in label])
        i1 += label_len

    return torch.stack(new_imgs), targets, torch.tensor(target_lengths)
