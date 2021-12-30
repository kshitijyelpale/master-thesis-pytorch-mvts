from multiprocessing import cpu_count
from tqdm.auto import tqdm

tqdm.pandas()
import numpy as np

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
import torchmetrics.functional as tmf

from utils import WINDOW_SIZE


class MotionDataset(Dataset):
    def __init__(self, dataset_sequences, regression=False):
        super(MotionDataset, self).__init__()
        self.sequences = dataset_sequences
        self.regression = regression

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence, label = self.sequences[idx]

        if self.regression is True:
            return dict(sequence=torch.Tensor(sequence.to_numpy()), label=torch.Tensor(label.to_numpy()))

        return dict(sequence=torch.Tensor(sequence.to_numpy()), label=torch.tensor(label).long())


class MotionDataModule(pl.LightningDataModule):
    def __init__(self, train_sequences, validation_sequences, test_sequences, batch_size, regression=False):
        super(MotionDataModule, self).__init__()
        self.train_sequences = train_sequences
        self.val_sequences = validation_sequences
        self.test_sequences = test_sequences
        self.batch_size = batch_size
        self.perform_regression = regression

    def setup(self, stage=None):
        self.train_dataset = MotionDataset(self.train_sequences, regression=self.perform_regression)
        self.val_dataset = MotionDataset(self.val_sequences, regression=self.perform_regression)
        self.test_dataset = MotionDataset(self.test_sequences, regression=self.perform_regression)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=cpu_count())

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=cpu_count())

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=cpu_count())


class LSTMModel(nn.Module):
    def __init__(self, n_features, n_classes, n_hidden=64, n_layers=5):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(input_size=n_features, hidden_size=n_hidden, num_layers=n_layers, batch_first=True,
                            dropout=0.5, bidirectional=False)

        self.classifier = nn.Linear(n_hidden, n_classes)

    def forward(self, x):
        # print(f'x shape {x.shape}')   # [128, 60, 33]
        self.lstm.flatten_parameters()
        _, (hidden, _) = self.lstm(x)
        # print(f'hidden shape after lstm {hidden.shape}')  # [10, 128, 64]

        out = hidden[-1]
        # print(f'out shape {out.shape}')  # [128, 64]
        out = self.classifier(out)
        # print(f'out shape after classifier {out.shape}')  # [128, 20]
        return out


class LSTMRegression(nn.Module):
    def __init__(self, n_features, n_hidden=64, n_layers=2):
        super(LSTMRegression, self).__init__()

        self.lstm = nn.LSTM(input_size=n_features, hidden_size=n_hidden, num_layers=n_layers, batch_first=True,
                            dropout=0.5, bidirectional=False)

        self.regressor = nn.Linear(n_hidden, n_features)

    def forward(self, x):
        # print(f'x shape {x.shape}')   # [128, 60, 33]
        self.lstm.flatten_parameters()
        x, (hidden, _) = self.lstm(x)
        # print(f'x shape after lstm {x.shape}')   # [128, 60, 64]
        # print(f'hidden shape after lstm {hidden.shape}')   # [2, 128, 64]

        out = hidden[-1]
        # print(f'out shape {out.shape}')  # [128, 64]
        output = self.regressor(out)
        # print(f'out shape after regressor {output.shape}')  # [128, 33]

        output_np = output.cpu().detach().numpy()
        output_new = np.repeat(output_np[:, np.newaxis, :], x.shape[1], axis=1)
        output = torch.tensor(output_new, requires_grad=True).to('cuda')
        # print(f'out shape after repeat {output.shape}')  # [128, 60, 33]

        return output


class GRUModel(nn.Module):
    def __init__(self, n_features, n_classes, n_hidden=64, n_layers=5):
        super(GRUModel, self).__init__()

        self.gru = nn.GRU(input_size=n_features, hidden_size=n_hidden, num_layers=n_layers, batch_first=True,
                          dropout=0.5)

        self.classifier = nn.Linear(n_hidden, n_classes)

    def forward(self, x):
        self.gru.flatten_parameters()
        _, hidden = self.gru(x)

        out = hidden[-1]
        return self.classifier(out)


class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(Encoder, self).__init__()

        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

        self.lstm1 = nn.LSTM(input_size=n_features, hidden_size=self.hidden_dim, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.embedding_dim, num_layers=1, batch_first=True)

    def forward(self, x):
        x, (_, _) = self.lstm1(x)
        x, (hidden_n, _) = self.lstm2(x)
        # hidden_n.reshape((self.n_features, self.embedding_dim))
        return hidden_n[-1]


class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim=64, output_dim=1):
        super(Decoder, self).__init__()

        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.output_dim = 2 * input_dim, output_dim

        self.lstm1 = nn.LSTM(input_size=input_dim, hidden_size=input_dim, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=input_dim, hidden_size=self.hidden_dim, num_layers=1, batch_first=True)

        self.dense_layers = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, x):
        x_np = x.cpu().detach().numpy()
        output_new = np.repeat(x_np[:, np.newaxis, :], self.seq_len, axis=1)
        x = torch.tensor(output_new, requires_grad=True).to('cuda')

        x, (_, _) = self.lstm1(x)
        x, (hidden, _) = self.lstm2(x)
        x = self.dense_layers(x)
        return x


class RAE(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(RAE, self).__init__()

        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim = embedding_dim

        self.encoder = Encoder(seq_len, n_features, embedding_dim)
        self.decoder = Decoder(seq_len, embedding_dim, n_features)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x


class TransformerModel(nn.Module):
    def __init__(self, n_features, n_classes):
        super(TransformerModel, self).__init__()

        self.layers = nn.TransformerEncoderLayer(d_model=n_features, nhead=1, batch_first=True, dropout=0.5)
        self.transformer = nn.TransformerEncoder(self.layers, num_layers=1)
        self.classifier = nn.Linear(n_features, n_classes)

    def forward(self, x):
        x = self.transformer(x)

        return self.classifier(x)


class LSTMClassifier(pl.LightningModule):
    def __init__(self, n_features: int, n_classes: int, classifier='LSTM'):
        super(LSTMClassifier, self).__init__()

        if classifier == 'Transformer':
            self.model = TransformerModel(n_features, n_classes)
        elif classifier == 'RAE':
            self.model = RAE(WINDOW_SIZE, n_features)
        elif classifier == 'GRU':
            self.model = GRUModel(n_features, n_classes)
        elif classifier == 'LSTMR':
            self.model = LSTMRegression(n_features)
        else:
            self.model = LSTMModel(n_features, n_classes)

        self.classifier = classifier

        if classifier in ['LSTMR', 'RAE']:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, labels=None):
        output = self.model(x)
        loss = 0

        if self.classifier in ['LSTMR', 'RAE']:
            loss = self.criterion(output, labels)
        elif labels is not None:
            if self.classifier == 'Transformer':
                labels = labels.unsqueeze(1).repeat_interleave(output.size(2), dim=1)

            loss = self.criterion(output, labels)

        return loss, output

    def training_step(self, batch, batch_idx):
        return self.log_loss_acc(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self.log_loss_acc(batch, 'val')

    def validation_epoch_end(self, outputs):
        # outputs = list of dictionaries
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}
        # use key 'log'
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        return self.log_loss_acc(batch, 'test')

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.005)

    def log_loss_acc(self, batch, name=''):
        sequences_to = batch["sequence"]
        labels = batch["label"]
        loss, output = self(sequences_to, labels)

        if self.classifier in ['LSTMR', 'RAE']:
            step_accuracy = 1 - loss
            self.log(f'{name}_loss', loss, prog_bar=True, logger=True)
            self.log(f'{name}_accuracy', step_accuracy, prog_bar=True, logger=True)

            return {"loss": loss}
        elif self.classifier == 'Transformer':
            labels = labels.unsqueeze(1).repeat_interleave(output.size(2), dim=1)

        predictions = torch.argmax(output, dim=1)
        step_accuracy = tmf.accuracy(predictions, labels)

        self.log(f'{name}_loss', loss, prog_bar=True, logger=True)
        self.log(f'{name}_accuracy', step_accuracy, prog_bar=True, logger=True)

        return {"loss": loss, "accuracy": step_accuracy}
