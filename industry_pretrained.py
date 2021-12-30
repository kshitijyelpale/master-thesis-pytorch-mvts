import gc
import sys
import time
import json

import pandas as pd
import numpy as np
from tqdm.auto import tqdm

tqdm.pandas()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torchmetrics.functional as tmf

from modules import MotionDataset, MotionDataModule, LSTMModel, LSTMClassifier
from utils import pre_process_data

############################
classifier = 'LSTM'
gpu_id = 3
partial_industry_data = None

perform_regression = False
if classifier in ['LSTMR', 'RAE']:
    perform_regression = True

results_to_file = 'exp_industry_pretraining_lstm_5_64.txt'
# results_to_file = 'exp_industry_pretraining_research_5_64.txt'

# seeds = [67720, 86618, 20207, 52756, 49845, 87547, 34495, 31572, 88824, 65435]
exp_result = {}
for i in range(10):
    seed = np.random.randint(100000)

    pl.seed_everything(seed)

    industry = pd.read_pickle("/hri/storage/user/kyelpale/industry_df.pkl")
    industry.progress_apply(lambda x: x)

    print(f'Data read successful')

    ###################### Pre-processing #################################

    # full_body_joints = ['pelvis', 'l3', 't8', 'neck', 'head', 'leftshoulder', 'rightshoulder', 'leftforearm',
    #                     'rightforearm', 'leftlowerleg', 'rightlowerleg']
    #
    # # full_body_joints = full_body_joints + ['lefttoe', 'righttoe', 'lefthand', 'righthand']
    #
    # # full_body_joints = full_body_joints + ['leftupperleg', 'rightupperleg', 'l5', 't12']
    #
    # used_columns = [col for col in industry.columns if col.startswith('position_')]
    # full_body_joints_columns = []
    # for col in used_columns:
    #     c = col.split('_')
    #     if c[1] in full_body_joints:
    #         full_body_joints_columns.append(col)
    # # used_columns.sort()
    # # used_columns = used_columns[0:33]
    # used_columns = full_body_joints_columns

    with open("columns_for_transfer_learning.txt", "r") as f:
        # json.dump(used_columns, f)
        used_columns = json.load(f)

    sequence_id_label = 'sequence_id_depos'
    label_column_name = 'label_depos'

    # drop_columns = ['en_label', 'subject_id', 'sequence_id_gepos', 'sequence_id_depos', 'sequence_id_cuact',
    #                 'label_gepos', 'label_depos', 'label_cuact']
    #
    # smf = [col for col in industry.columns if col.startswith('sensormagneticfield')]
    # drop_columns = drop_columns + smf

    train_sequences, val_sequences, test_sequences, label_encoder = pre_process_data(industry, used_columns,
                                                                                     sequence_id_label,
                                                                                     label_column_name,
                                                                                     partial_sequences=partial_industry_data,
                                                                                     regression=perform_regression
                                                                                     )

    ###################### Dataset #################################

    N_EPOCHS = 100  # 250
    BATCH_SIZE = 2048

    # data_module = MotionDataModule(train_sequences, val_sequences, test_sequences, BATCH_SIZE)

    ###################### Model #################################

    for j in range(1):
        j = j + 10
        # k = k + 1
        # if k == 6:
        #     j = 7.5
        # elif k == 7:
        #     j = 10
        # else:
        #     j = k

        train_sequences_subset = train_sequences[0:int(len(train_sequences) * (j / 10))]

        data_module = MotionDataModule(train_sequences_subset, val_sequences, test_sequences, BATCH_SIZE,
                                       regression=perform_regression)

        model = LSTMClassifier(n_features=len(used_columns), n_classes=len(label_encoder.classes_),
                               classifier=classifier)

        checkpoints_path = '/hri/storage/user/kyelpale/Tmp/checkpoints'
        checkpoint_callback = ModelCheckpoint(dirpath=checkpoints_path, filename="industry", save_top_k=1, verbose=True,
                                              monitor="val_loss", mode="min")

        early_stop_callback = EarlyStopping(monitor="val_accuracy", min_delta=0.00, patience=30, verbose=False,
                                            mode="max")

        logger = TensorBoardLogger("/hri/storage/user/kyelpale/lightning_logs", name="industry")

        # os.system('tensorboard --logdir=/hri/storage/user/kyelpale/phume/lightning-logs')

        trainer = pl.Trainer(max_epochs=N_EPOCHS, fast_dev_run=False, log_every_n_steps=1, auto_lr_find=True,
                             logger=logger, checkpoint_callback=True,
                             callbacks=[checkpoint_callback, early_stop_callback],
                             progress_bar_refresh_rate=30, gpus=[gpu_id])

        print(f'Model training started')
        trainer.fit(model, data_module)

        test_result = trainer.test()
        if trainer.root_gpu in [0, 1, 2, 3]:
            print(f'test accuracy: {test_result[0]["test_accuracy"]}')
            # print(trainer.gpus, trainer.num_gpus, trainer.root_gpu)

            accuracy = test_result[0]["test_accuracy"]

            # if not exp_result.get(seed):
            #     exp_result[seed] = {}
            # exp_result[seed][j / 10] = accuracy

            exp_result[seed] = (accuracy, trainer.current_epoch, trainer.checkpoint_callback.best_model_path)
            print(exp_result)

            with open(results_to_file, "a") as f:
                json.dump(exp_result, f)
                f.write('\n')

    gc.collect()

sys.exit(0)

################# Predictions ###################

print(f'Prediction on test sequences')
# trainer.checkpoint_callback.best_model_path    checkpoints_path + '/industry.ckpt'
trained_model = LSTMClassifier.load_from_checkpoint(trainer.checkpoint_callback.best_model_path,
                                                    n_features=len(used_columns.columns),
                                                    n_classes=len(label_encoder.classes_))

trained_model.freeze()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trained_model.to(device)

test_dataset = MotionDataset(test_sequences)

if trainer.root_gpu == 0:

    predictions = []
    labels = []

    start = time.time()

    for item in tqdm(test_dataset):
        sequence = item["sequence"].to(device)
        label = item["label"].to(device)

        _, output = trained_model(sequence.unsqueeze(dim=0))
        prediction = torch.argmax(output, dim=1)
        predictions.append(prediction.item())
        labels.append(label.item())

    print(f'Time for testing: {time.time() - start} seconds')

    print(f'accuracy: {tmf.accuracy(torch.IntTensor(predictions), torch.IntTensor(labels))}')
    print(classification_report(labels, predictions, target_names=label_encoder.classes_))

# cm = confusion_matrix(labels, predictions)
# df_cm = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)
# show_confusion_matrix(df_cm)


# nest_cols = {}
# for col in industry.columns.to_list():
#     x = col.split('_')
#     if not nest_cols.get(x[0]):
#         nest_cols[x[0]] = set()
#
#     nest_cols[x[0]].add(x[1])
