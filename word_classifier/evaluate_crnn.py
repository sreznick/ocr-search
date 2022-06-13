import os
from functools import partial
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T
from word_recognition import crnn, dataset_utils, decoders
import shared_paths as paths


BATCH_SIZE = 128
DATASET_DIR = os.path.join(paths.DATA_DIR, 'words', 'test')


@torch.no_grad()
def main():
    # load test dataset
    dataset = dataset_utils.WordDataset(
        root=DATASET_DIR,
        transform=T.Compose((
            T.Grayscale(1),
            T.ToTensor(),
            T.functional.invert
        ))
    )

    # load tools for encoding and decoding sequences
    # itos: tokens (int) -> characters (str)
    # stoi: characters -> tokens
    with open(os.path.join(paths.OUTPUT_DIR, 'chars.txt'), 'r') as f:
        itos = f.read().split('\n')[:-1]
    stoi = dict((ch, i) for i, ch in enumerate(itos))

    # collate_fn pads input for batching
    collate_fn = partial(dataset_utils.collate_fn, vocab=stoi)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn
    )

    # load trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = crnn.CRNN(output_dim=len(itos))
    model.load_state_dict(
        torch.load(os.path.join(paths.OUTPUT_DIR, 'model.pth')))
    model.to(device)

    # evaluate on test set
    loss_fn = torch.nn.CTCLoss()
    total_loss = 0.0  # weighted sum of batch losses
    hits = 0          # number of accurate word predictions
    for batch in dataloader:
        # forward pass
        X, y, len_y = (t.to(device) for t in batch)
        logits = model(X)
        log_probs = F.log_softmax(logits, dim=-1)
        len_X = torch.LongTensor([log_probs.size(0)] * len(X))
        loss = loss_fn(log_probs, y, len_X, len_y)
        total_loss += loss.item() * len(X)

        # decode output
        log_probs = log_probs.cpu().transpose(1, 0).numpy()
        predictions = [decoders.ctc_greedy(logp, itos)
                       for logp in log_probs]

        # decode target sequences (words)
        i1 = 0
        targets = []
        for len_yi in len_y:
            word = ''.join([itos[y[i1+delta]] for delta in range(len_yi)])
            targets.append(word)
            i1 += len_yi

        # count accurate predictions
        hits += sum(1 if p == t else 0 for p, t in zip(predictions, targets))

    print(f'Test loss: {total_loss / len(dataset):.2e}')
    print(f'Test accuracy: {hits / len(dataset) * 100:.1f}%')


if __name__ == '__main__':
    main()
