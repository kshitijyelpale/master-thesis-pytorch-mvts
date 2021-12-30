import gc
import json
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

tqdm.pandas()
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torchmetrics.functional as tmf

from modules import MotionDataset, MotionDataModule, LSTMModel, LSTMClassifier
from utils import prepare_sequences, show_confusion_matrix, pre_process_data

##################################

exp_result = {}
classifier = 'LSTM'
gpu_id = 2

perform_regression = False
if classifier in ['LSTMR', 'RAE']:
    perform_regression = True

results_to_file = "exp_phume_train_test_lstm_5.txt"
# results_to_file = "tmp.txt"

for i in range(10):
    seed = np.random.randint(100000)
    pl.seed_everything(seed)

    phume = pd.read_pickle("/hri/storage/user/kyelpale/data_frame.pkl")
    phume.progress_apply(lambda x: x)
    # data_to_drop = np.random.choice(phume.shape[0], size=int(phume.shape[0] * .99), replace=False)
    # phume.drop(data_to_drop, axis=0, inplace=True)

    print(f'Data read successful')

    ###################### Pre-processing #################################

    with open("columns_for_transfer_learning.txt", "r") as f:
        used_columns = json.load(f)

    sequence_id_label = 'sequence_id'
    label_column_name = 'label'
    train_sequences, val_sequences, test_sequences, label_encoder = pre_process_data(phume, used_columns,
                                                                                     sequence_id_label,
                                                                                     label_column_name,
                                                                                     regression=perform_regression)

    ###################### Percentage Data training #################################

    N_EPOCHS = 100  # 250
    BATCH_SIZE = 128

    for j in range(10):
        j = j + 1
        train_sequences_subset = train_sequences[0:int(len(train_sequences) * (j / 10))]

        data_module = MotionDataModule(train_sequences_subset, val_sequences, test_sequences, BATCH_SIZE,
                                       regression=perform_regression)

        model = LSTMClassifier(n_features=len(used_columns), n_classes=len(label_encoder.classes_),
                               classifier=classifier)

        checkpoints_path = '/hri/storage/user/kyelpale/Tmp/checkpoints'
        checkpoint_callback = ModelCheckpoint(dirpath=checkpoints_path, filename="phume", save_top_k=1, verbose=True,
                                              monitor="val_loss", mode="min")

        early_stop_callback = EarlyStopping(monitor="val_accuracy", min_delta=0.00, patience=30, verbose=False,
                                            mode="max")

        logger = TensorBoardLogger("/hri/storage/user/kyelpale/lightning_logs", name="phume")

        # os.system('tensorboard --logdir=/hri/storage/user/kyelpale/phume/lightning-logs')

        trainer = pl.Trainer(max_epochs=N_EPOCHS, fast_dev_run=False, log_every_n_steps=1, auto_lr_find=True,
                             checkpoint_callback=True, callbacks=[checkpoint_callback, early_stop_callback],
                             progress_bar_refresh_rate=30, logger=logger, gpus=[gpu_id])

        print(f'Model training started')
        trainer.fit(model, data_module)

        test_result = trainer.test()
        if trainer.root_gpu in [0, 1, 2, 3]:
            print(f'test accuracy: {test_result[0]["test_accuracy"]}')
            accuracy = test_result[0]["test_accuracy"]

            if not exp_result.get(seed):
                exp_result[seed] = {}
            exp_result[seed][j / 10] = accuracy
            # exp_result[seed] = (accuracy, trainer.checkpoint_callback.best_model_path)
            print(exp_result)

            with open(results_to_file, "a") as f:
                json.dump(exp_result, f)
                f.write('\n')

    gc.collect()

    # data_percent_drop = round(data_percent_drop - 0.1, 1)
    # data_to_drop = np.random.choice(phume.shape[0], size=int(phume.shape[0] * data_percent_drop), replace=False)
    # phume.drop(data_to_drop, axis=0, inplace=True)

    # phume.label.value_counts().plot(kind="bar")
    # plt.xticks(rotation=45)
    # plt.show()

    # nest_cols = {}
    # for col in phume.columns.to_list():
    #     x = col.split('_')
    #     if not nest_cols.get(x[0]):
    #         nest_cols[x[0]] = set()
    #
    #     nest_cols[x[0]].add(x[1])

    ################# Predictions ###################

    # print(f'Prediction on test sequences')
    # trained_model = LSTMClassifier.load_from_checkpoint(trainer.checkpoint_callback.best_model_path,
    #                                                     n_features=len(used_columns),
    #                                                     n_classes=len(label_encoder.classes_))
    #
    # trained_model.freeze()
    #
    # test_dataset = MotionDataset(test_sequences)
    #
    # predictions = []
    # labels = []
    #
    # for item in tqdm(test_dataset):
    #     sequence = item["sequence"]
    #     label = item["label"]
    #
    #     _, output = trained_model(sequence.unsqueeze(dim=0))
    #     prediction = torch.argmax(output, dim=1)
    #     predictions.append(prediction.item())
    #     labels.append(label.item())
    #
    # accuracy = tmf.accuracy(torch.IntTensor(predictions), torch.IntTensor(labels))
    # print(f'accuracy: {accuracy}')
    # print(classification_report(labels, predictions, target_names=label_encoder.classes_))

    # cm = confusion_matrix(labels, predictions)
    # df_cm = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)
    # show_confusion_matrix(df_cm)
