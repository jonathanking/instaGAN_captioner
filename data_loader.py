import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from build_vocab import Vocabulary
import regex

START = "\\"
END = "\n"


def add_START_END(cap):
    """ Prepares a raw caption for input to the model. """
    cap = START + cap + END
    return cap


class InstgramDataset(data.Dataset):
    """Instagram Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, metadata_path, images_path, vocab, transform=None):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            root: image directory.
            json: annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        with open(metadata_path, "rb") as md:
            self.captions = list(map(add_START_END, pickle.load(md)))
        self.images = torch.load(images_path)
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        caption_str = self.captions[index]
        image = self.images[index]
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        caption = [self.vocab(token) for token in caption_str]
        caption = torch.Tensor(caption)
        return image, caption

    def __len__(self):
        return self.images.shape[0]


class CaptionDataset(data.Dataset):
    """Instagram Caption Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, vocab, seq_len, from_file=False):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            root: caption file.
            vocab: vocabulary wrapper.
        """
        if not from_file:
            with open(root, "r") as f:
                text = f.read()
            text_as_int = torch.tensor(np.array([vocab.char2idx[c] for c in text]))
        else:
            text_as_int = torch.load(root)

        self.captions = []
        self.targets = []
        for i in range(len(text_as_int) // seq_len - seq_len):
            start_idx = i * seq_len
            end_idx = start_idx + seq_len
            self.captions.append(text_as_int[start_idx:end_idx])
            self.targets.append(text_as_int[start_idx + 1:end_idx + 1])
        self.root = root
        self.vocab = vocab

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        return self.captions[index], self.targets[index ]

    def __len__(self):
        return len(self.captions)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption).
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, lengths


def collate_fn_captions(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption).
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    captions_src, captions_tgt = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    # images = torch.stack(images, 0)

    # Merge captions_src (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions_src]
    captions_src_mrg = torch.zeros(len(captions_src), max(lengths)).long()
    for i, cap in enumerate(captions_src):
        end = lengths[i]
        captions_src_mrg[i, :end] = cap[:end]

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions_tgt]
    captions_tgt_mrg = torch.zeros(len(captions_tgt), max(lengths)).long()
    for i, cap in enumerate(captions_tgt):
        end = lengths[i]
        captions_tgt_mrg[i, :end] = cap[:end]
    return captions_src_mrg, captions_tgt_mrg, lengths


def get_loader(metadata_path, images_path, vocab, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    insta = InstgramDataset(metadata_path=metadata_path,
                       images_path=images_path,
                       vocab=vocab,
                       transform=transform)
    
    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=insta,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader


def get_caption_loader(root, vocab, batch_size, shuffle, num_workers, seq_len=100):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    insta = CaptionDataset(root=root,
                            vocab=vocab,
                            seq_len=seq_len, from_file=True)

    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=insta,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader