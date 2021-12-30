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
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torchmetrics.functional as tmf

from modules import MotionDataset, MotionDataModule, LSTMModel, LSTMClassifier
from utils import pre_process_data

############################
exp_result = {}
classifier = 'LSTM'
gpu_id = 1
partial_industry_data = None
results_to_file = "exp_industry_fine_tuning_lstm_8.txt"

# checkpoint_names = ['2633', '2639', '2643', '2649', '2654', '2660', '2665', '2670', '2675', '2678']  # 4 lstm 33 columns
# checkpoint_names = ['2632', '2637', '2642', '2648', '2652', '2656', '2662', '2667', '2671', '2677']  # 5 lstm 33 columns
# checkpoint_names = ['2634', '2638', '2644', '2647', '2651', '2655', '2659', '2664', '2668', '2673']  # 6 lstm 33 columns
# checkpoint_names = ['2635', '2640', '2645', '2650', '2657', '2661', '2666', '2672', '2676', '2680']  # 7 lstm 33 columns
checkpoint_names = ['2636', '2641', '2646', '2653', '2658', '2663', '2669', '2674', '2679', '2681']  # 8 lstm 33 columns

for i in range(10):
    phume_checkpoint = f'/phume-v{checkpoint_names[i]}.ckpt'
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
                                                                                     partial_sequences=partial_industry_data
                                                                                     )

    ###################### Dataset #################################

    N_EPOCHS = 100  # 250
    BATCH_SIZE = 2048

    # data_module = MotionDataModule(train_sequences, val_sequences, test_sequences, BATCH_SIZE)

    ###################### Model #################################

    for k in range(7):
        k = k + 1
        if k == 6:
            j = 7.5
        elif k == 7:
            j = 10
        else:
            j = k
        train_sequences_subset = train_sequences[0:int(len(train_sequences) * (j / 10))]
        # model = LSTMClassifier(n_features=len(used_columns), n_classes=len(label_encoder.classes_), classifier=classifier)
        data_module = MotionDataModule(train_sequences_subset, val_sequences, test_sequences, BATCH_SIZE)
        checkpoints_path = '/hri/storage/user/kyelpale/Tmp/checkpoints'
        pre_trained_model = LSTMClassifier.load_from_checkpoint(checkpoints_path + phume_checkpoint,
                                                                n_features=len(used_columns), n_classes=20,
                                                                classifier=classifier)

        # changing last layer with phume dataset labels
        hidden_units = pre_trained_model.model.classifier.in_features
        # Parameters of newly constructed last layer has requires_grad=True by default
        pre_trained_model.model.classifier = nn.Linear(hidden_units, len(label_encoder.classes_))

        checkpoint_callback = ModelCheckpoint(dirpath=checkpoints_path, filename="fine_tune_industry", save_top_k=1,
                                              verbose=True,
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
        trainer.fit(pre_trained_model, data_module)

        test_result = trainer.test()
        if trainer.root_gpu in [0, 1, 2, 3]:
            print(f'test accuracy: {test_result[0]["test_accuracy"]}')
            # print(trainer.gpus, trainer.num_gpus, trainer.root_gpu)

            accuracy = test_result[0]["test_accuracy"]

            if not exp_result.get(seed):
                exp_result[seed] = {}
            exp_result[seed][j / 10] = accuracy
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
