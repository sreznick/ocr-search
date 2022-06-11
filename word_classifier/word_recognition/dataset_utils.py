from typing import Optional, Callable, Iterable
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

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
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


def make_char_vocab(dataset: Dataset) -> tt.vocab.Vocab:
    chars = set()
    for _, label, _ in dataset:
        for ch in label:
            chars.add(ch)
    return tt.vocab.build_vocab_from_iterator(
        chars, specials=['<->'], special_first=True)


def decode(vocab: tt.vocab.Vocab, tokens: Iterable[int]) -> str:
    """
    Convert a sequence of tokens into a string.
    """
    itos = vocab.get_itos()
    return ''.join(itos[ti] for ti in tokens)


def collate_fn(batch: list[tuple], vocab: tt.vocab.Vocab
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    imgs, labels, target_lengths = zip(*batch)

    # pad images with zeros
    nc, h, _ = imgs[0].shape
    img_widths = [img.size(-1) for img in imgs]
    w = max(img_widths)
    if all(w_i == w for w_i in img_widths):
        new_imgs = imgs
    else:
        new_imgs = []
        for img in imgs:
            new_img = torch.zeros(nc, h, w)
            w_i = img.size(2)
            new_img[:, :, :w_i] = img
            new_imgs.append(new_img)

    # convert labels to tensors of tokens
    targets = torch.zeros(sum(target_lengths), dtype=torch.int64)
    i1 = 0
    for label, label_len in zip(labels, target_lengths):
        targets[i1:i1+label_len] = torch.tensor([vocab[ch] for ch in label])
        i1 += label_len

    return torch.stack(new_imgs), targets, torch.tensor(target_lengths)
