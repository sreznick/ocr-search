import os
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn import CTCLoss
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms as T
import torchtext.vocab as tvcb
import shared_paths as paths
from word_recognition import dataset_utils, crnn


DATASET_DIR = os.path.join(paths.DATA_DIR, 'words', 'train')
BATCH_TRAIN = 64
BATCH_EVAL = 128
LR = 1e-3
GRAD_CLIP = 1.0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VAL_SIZE = 0.1
ITERATIONS = 60000
EVAL_EVERY = 2000


def main():
    print(f'Using {DEVICE} device')

    # load dataset
    dataset = dataset_utils.WordDataset(
        root=DATASET_DIR,
        transform=T.Compose((
            T.Grayscale(1),
            T.ToTensor(),
            T.functional.invert
        ))
    )

    # split dataset into train and validate and test
    val_size = int(round(len(dataset) * VAL_SIZE))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # build vocab for encoding/decoding
    chars = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
    chars = chars + chars.upper()
    vocab = tvcb.build_vocab_from_iterator(chars, specials=['<->'],
                                           special_first=True)
    with open(os.path.join(paths.OUTPUT_DIR, 'chars.txt'), 'w') as f:
        for character in vocab.get_itos():
            f.write(character + '\n')

    # create dataloaders
    # collate_fn takes care of different input sizes
    collate_fn = partial(dataset_utils.collate_fn, vocab=vocab)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_TRAIN,
        collate_fn=collate_fn,
        shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_EVAL,
        collate_fn=collate_fn
    )

    # train the model
    model = crnn.CRNN(output_dim=len(vocab)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = CTCLoss(blank=0)
    train_loss, val_loss = train(model, train_dataloader, val_dataloader,
                                 optimizer, loss_fn)

    plt.figure()
    x = EVAL_EVERY * np.arange(1, len(train_loss) + 1)
    plt.plot(x, train_loss, marker='.', label='train')
    plt.plot(x, val_loss, marker='.', label='validate')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(paths.OUTPUT_DIR, 'CRNN_loss.png'), dpi=150)


def train(model, train_dataloader, val_dataloader, optimizer, loss_fn,
          total_iterations=ITERATIONS, eval_every=EVAL_EVERY,
          save_to=paths.OUTPUT_DIR, device=DEVICE):
    model.train()
    train_loss = []
    val_loss = []
    num_iter = 0  # number of model parameters updates
    cur_loss, cur_samples = 0.0, 0  # loss = cur_loss / cur_samples
    while True:
        for batch in train_dataloader:
            # single train iteration
            loss, bs = forward_pass(model, batch, loss_fn, device)
            optimizer.zero_grad()
            loss.backward()
            if GRAD_CLIP:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            # update current loss
            cur_loss += loss.item() * bs
            cur_samples += bs
            num_iter += 1

            # evaluate model
            if num_iter % eval_every == 0:
                train_loss.append(cur_loss / cur_samples)
                print(f'[{num_iter} iterations] ' +
                      f'train loss: {train_loss[-1]:.2e}', end=' ')
                cur_loss, cur_samples = 0.0, 0
                model.eval()
                with torch.no_grad():
                    for batch in val_dataloader:
                        loss, bs = forward_pass(model, batch, loss_fn, device)
                        cur_loss += loss.item() * bs
                        cur_samples += bs
                    val_loss.append(cur_loss / cur_samples)
                    print(f'val loss: {val_loss[-1]:.2e}')
                    cur_loss, cur_samples = 0.0, 0

                    # save current weights if achieved lowest validation loss
                    if np.argmin(val_loss) == len(val_loss) - 1:
                        torch.save(model.state_dict(),
                                   os.path.join(save_to, 'model.pth'))
                model.train()

            if num_iter == total_iterations:
                return train_loss, val_loss


def forward_pass(model, batch, loss_fn, device):
    X, y, len_y = (t.to(device) for t in batch)
    batch_size = len(X)
    logits = model(X)
    log_probs = F.log_softmax(logits, dim=-1)
    len_X = torch.LongTensor([log_probs.size(0)] * batch_size)
    loss = loss_fn(log_probs, y, len_X, len_y)
    return loss, batch_size


if __name__ == '__main__':
    main()
