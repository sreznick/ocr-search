import torch
from torch import nn
import word_recognition.feature_extraction as fe


class CRNN(nn.Module):
    def __init__(self, output_dim: int, cnn_hidden: int = 64,
                 rnn_hidden: int = 256):
        super().__init__()
        self.cnn = fe.OriginalCNN()
        self.feat2seq = fe.Features2Seq(in_channels=512, img_height=1,
                                        out_features=64)
        self.rnn1 = nn.LSTM(input_size=cnn_hidden, hidden_size=rnn_hidden,
                            bidirectional=True)
        self.rnn2 = nn.LSTM(input_size=rnn_hidden*2, hidden_size=rnn_hidden,
                            bidirectional=True)
        self.fc = nn.Linear(rnn_hidden*2, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x)
        seq = self.feat2seq(features)
        out, _ = self.rnn2(self.rnn1(seq)[0])
        return self.fc(out)
