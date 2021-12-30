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
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torchmetrics.functional as tmf

from modules import MotionDataset, MotionDataModule, LSTMModel, LSTMClassifier
from utils import prepare_sequences, show_confusion_matrix, pre_process_data

##################################

exp_result = {}
classifier = 'LSTM'
gpu_id = 0
results_to_file = "exp_phume_fine_tuning_lstm_5.txt"

# checkpoint_names = ['197']  # 5 lstm 33 columns # '181', '183', '186', '189', '192', '195', '196', '197', '198', '199'
# checkpoint_names = ['1473', '1474', '1475', '1476', '1477', '1478', '1479', '1480', '1481', '1482']   # 5 lstm 33 columns repeat
# checkpoint_names = ['1483', '1484', '1485', '1486', '1487', '1488', '1489', '1490', '1491', '1492']
# checkpoint_names = ['1513', '1515', '1517', '1521', '1524', '1526', '1530', '1532', '1535', '1538']
# checkpoint_names = ['1514', '1516', '1519', '1522', '1525', '1528', '1531', '1534', '1537', '1539']
checkpoint_names = ['1512', '1518', '1520', '1523', '1527', '1529', '1533', '1536', '1540', '1541']

# checkpoint_names = ['1502', '1503', '1504', '1505', '1506', '1507', '1508', '1509', '1510', '1511']   # 5 lstm 33 columns bidi
# checkpoint_names = ['80']  # 2 lstm 33 columns  '80', '82', '84', '87', '89', '92', '94', '96', '98', '99'

# checkpoint_names = ['1406', '1409', '1413', '1414', '1417', '1419', '1420', '1421', '1423', '1425']  # 5 lstm 33 columns partial indus 5%
# checkpoint_names = ['1407', '1412', '1415', '1418', '1424', '1427', '1429', '1431', '1432', '1433']  # 5 lstm 33 columns partial indus 10%
# checkpoint_names = ['1408', '1416', '1426', '1430', '1434', '1437', '1440', '1441', '1444', '1445']  # 5 lstm 33 columns partial indus 25%
# checkpoint_names = ['1410', '1422', '1435', '1438', '1442', '1446', '1448', '1450', '1451', '1453']  # 5 lstm 33 columns partial indus 50%
# checkpoint_names = ['1411', '1428', '1436', '1439', '1443', '1447', '1449', '1452', '1454', '1455']  # 5 lstm 33 columns partial indus 75%

for i in range(10):
    industry_checkpoint = f'/industry-v{checkpoint_names[i]}.ckpt'
    seed = np.random.randint(100000)
    pl.seed_everything(seed)

    phume = pd.read_pickle("/hri/storage/user/kyelpale/data_frame.pkl")
    phume.progress_apply(lambda x: x)

    print(f'Data read successful')

    ###################### Pre-processing #################################

    with open("columns_for_transfer_learning.txt", "r") as f:
        used_columns = json.load(f)

    sequence_id_label = 'sequence_id'
    label_column_name = 'label'
    train_sequences, val_sequences, test_sequences, label_encoder = pre_process_data(phume, used_columns,
                                                                                     sequence_id_label,
                                                                                     label_column_name)

    ######################## model fetch and training ########

    N_EPOCHS = 100  # 250
    BATCH_SIZE = 128

    for j in range(10):
        j = j + 1
        train_sequences_subset = train_sequences[0:int(len(train_sequences) * (j / 10))]

        data_module = MotionDataModule(train_sequences_subset, val_sequences, test_sequences, BATCH_SIZE)

        checkpoints_path = '/hri/storage/user/kyelpale/Tmp/checkpoints'
        pre_trained_model = LSTMClassifier.load_from_checkpoint(checkpoints_path + industry_checkpoint,
                                                                n_features=len(used_columns), n_classes=12,
                                                                classifier=classifier)

        # fix/freeze the deep neural network like LSTM, GRU parameters
        # for param in pre_trained_model.model.parameters():
        #     param.requires_grad = False

        # freeze only first 3 layers parameters
        # for p in range(3):
        #     for param in pre_trained_model.model.lstm.all_weights[p]:
        #         param.requires_grad = False

        # changing last layer with phume dataset labels
        hidden_units = pre_trained_model.model.classifier.in_features
        # Parameters of newly constructed last layer has requires_grad=True by default
        pre_trained_model.model.classifier = nn.Linear(hidden_units, len(label_encoder.classes_))

        checkpoint_callback = ModelCheckpoint(dirpath=checkpoints_path, filename="fine_tune_phume", save_top_k=1,
                                              verbose=True, monitor="val_loss", mode="min")

        early_stop_callback = EarlyStopping(monitor="val_accuracy", min_delta=0.00, patience=30, verbose=False,
                                            mode="max")

        logger = TensorBoardLogger("/hri/storage/user/kyelpale/lightning_logs", name="phume_fine_tuning")

        trainer = pl.Trainer(max_epochs=N_EPOCHS, fast_dev_run=False, log_every_n_steps=1, auto_lr_find=True,
                             checkpoint_callback=True, callbacks=[checkpoint_callback, early_stop_callback],
                             progress_bar_refresh_rate=30, logger=logger, gpus=[gpu_id])

        print(f'Model fine tuning started')
        trainer.fit(pre_trained_model, data_module)

        test_result = trainer.test()
        if trainer.root_gpu in [0, 1, 2, 3]:
            print(f'test accuracy: {test_result[0]["test_accuracy"]}')
            accuracy = test_result[0]["test_accuracy"]

            if not exp_result.get(seed):
                exp_result[seed] = {}
            exp_result[seed][j / 10] = accuracy
            print(exp_result)

            with open(results_to_file, "a") as f:
                json.dump(exp_result, f)
                f.write('\n')

    gc.collect()

################# Predictions ###################

# print(f'Prediction on test sequences')
# fine_tuned_model = LSTMClassifier.load_from_checkpoint(trainer.checkpoint_callback.best_model_path,
#                                                        n_features=len(used_columns.columns),
#                                                        n_classes=len(label_encoder.classes_))
#
# fine_tuned_model.freeze()
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
#     _, output = fine_tuned_model(sequence.unsqueeze(dim=0))
#     prediction = torch.argmax(output, dim=1)
#     predictions.append(prediction.item())
#     labels.append(label.item())
#
# accuracy = tmf.accuracy(torch.IntTensor(predictions), torch.IntTensor(labels))
# print(f'accuracy: {accuracy}')
# print(classification_report(labels, predictions, target_names=label_encoder.classes_))


# checkpoint_names = ['71', '72', '73']  # gru 60-columns '59', '61', '64', '66', '68', '69', '70',
# checkpoint_names = ['58', '60', '63', '65', '74'] # lstm 60-columns
# checkpoint_names = ['79']  # lstm 60-columns ['75', '76', '77', '78', '79']
# checkpoint_names = ['80', '82', '84', '87', '89', '92', '94', '96', '98', '99']  # 2 lstm 33 columns
# checkpoint_names = ['81', '83', '85', '86', '88', '90', '91', '93', '95', '97']  # 2 gru 33 columns
# checkpoint_names = ['101', '112', '118', '124', '131', '137', '144', '149', '153', '156']  # 4 lstm 33 columns
# checkpoint_names = ['100', '106', '110', '116', '119', '125', '130', '135', '140', '145']  # 4 gru 33 columns
# checkpoint_names = ['157']  # ['102', '111', '121', '129', '139', '150', '155', '157', '158', '159']  # 6 lstm 33 columns
# checkpoint_names = ['103', '108', '115', '123', '128', '134', '142', '147', '152', '154']  # 6 gru 33 columns
# checkpoint_names = ['105', '107', '113', '117', '122', '127', '132', '136', '141', '146']  # 8 lstm 33 columns
# checkpoint_names = ['138']  # ['104', '109', '114', '120', '126', '133', '138', '143', '148', '151']  # 8 gru 33 columns
# checkpoint_names = ['181', '183', '186', '189', '192', '195', '196', '197', '198', '199']  # 5 lstm 33 columns
# checkpoint_names = ['180', '182', '184', '185', '187', '188', '190', '191', '193', '194']  # 5 gru 33 columns
# checkpoint_names = ['200', '203', '208', '211', '214', '215', '216', '217', '218', '219']  # 7 lstm 33 columns
# checkpoint_names = ['201', '202', '204', '205', '206', '207', '209', '210', '212', '213']  # 7 gru 33 columns

# checkpoint_names = ['220', '224', '227', '234', '237', '241', '244', '247', '251', '252']  # 2 gru 27 columns
# checkpoint_names = ['221', '225', '228', '230', '233', '236', '240', '242', '246', '249']  # 2 lstm 27 columns
# checkpoint_names = ['223', '226', '231', '235', '239', '245', '248', '253', '255', '257']  # 4 gru 27 columns
# checkpoint_names = ['222', '229', '232', '238', '243', '250', '254', '256', '258', '259']  # 4 lstm 27 columns
# checkpoint_names = ['261', '263', '266', '268', '270', '272', '274', '277', '278', '279'] # 6 lstm 27 columns
# checkpoint_names = ['260', '262', '264', '265', '267', '269', '271', '273', '275', '276']  # 6 gru 27 columns
# checkpoint_names = ['281', '282', '284', '285', '288', '290', '292', '294', '296', '297']  # 5 gru 27 columns
# checkpoint_names = ['280', '283', '286', '287', '289', '291', '293', '295', '298', '299']  # 5 lstm 27 columns

# checkpoint_names = ['300', '304', '307', '311', '317', '320', '323', '327', '331', '339']  # 5 lstm 27 columns partial indus 5%
# checkpoint_names = ['301', '306', '310', '315', '321', '329', '334', '341', '343', '346']  # 5 lstm 27 columns partial indus 10%
# checkpoint_names = ['302', '313', '335', '347', '355', '361', '367', '374', '377', '381']  # 5 lstm 27 columns partial indus 25%
# checkpoint_names = ['303', '336', '351', '358', '366', '378', '384', '386', '389', '391']  # 5 lstm 27 columns partial indus 50%
# checkpoint_names = ['305', '337', '348', '356', '362', '372', '376', '380', '385', '387']  # 5 lstm 27 columns partial indus 75%

# checkpoint_names = ['308', '312', '316', '318', '322', '324', '326', '330', '333', '338']  # 5 gru 27 columns partial indus 5%
# checkpoint_names = ['309', '314', '319', '325', '328', '332', '340', '342', '344', '345']  # 5 gru 27 columns partial indus 10%
# checkpoint_names = ['349', '352', '353', '357', '359', '363', '365', '369', '370', '373']  # 5 gru 27 columns partial indus 25%
# checkpoint_names = ['350', '354', '360', '364', '368', '371', '375', '379', '382', '383']  # 5 gru 27 columns partial indus 50%
# checkpoint_names = ['388', '390', '392', '393', '394', '395', '396', '397', '398', '399']  # 5 gru 27 columns partial indus 75%

# checkpoint_names = ['411', '417', '424', '427', '435', '438', '444', '449', '453', '458']  # 4 lstm 27 columns partial indus 5%
# checkpoint_names = ['412', '419', '423', '436', '442', '446', '450', '460', '462', '468']  # 4 lstm 27 columns partial indus 10%
# checkpoint_names = ['413', '437', '447', '455', '471', '475', '476', '481', '484', '488']  # 4 lstm 27 columns partial indus 25%
# checkpoint_names = ['410', '429', '443', '456', '465', '473', '477', '482', '487', '493']  # 4 lstm 27 columns partial indus 50%
# checkpoint_names = ['409', '426', '454', '466', '474', '479', '485', '491', '495', '498']  # 4 lstm 27 columns partial indus 75%

# checkpoint_names = ['400', '402', '405', '414', '416', '421', '425', '431', '434', '440']  # 4 gru 27 columns partial indus 5%
# checkpoint_names = ['401', '403', '408', '415', '418', '422', '428', '433', '439', '445']  # 4 gru 27 columns partial indus 10%
# checkpoint_names = ['404', '420', '432', '448', '452', '459', '461', '463', '467', '469']  # 4 gru 27 columns partial indus 25%
# checkpoint_names = ['406', '441', '451', '464', '472', '480', '483', '489', '492', '494']  # 4 gru 27 columns partial indus 50%
# checkpoint_names = ['407', '430', '457', '470', '478', '486', '490', '496', '497', '499']  # 4 gru 27 columns partial indus 75%

# checkpoint_names = ['500', '504', '514', '517', '520', '527', '535', '542', '548', '557']  # 2 lstm 27 columns partial indus 5%
# checkpoint_names = ['501', '512', '518', '528', '540', '547', '558', '562', '566', '569']  # 2 lstm 27 columns partial indus 10%
# checkpoint_names = ['502', '522', '536', '550', '563', '574', '580', '585', '590', '593']  # 2 lstm 27 columns partial indus 25%
# checkpoint_names = ['503', '524', '544', '560', '570', '577', '581', '588', '594', '596']  # 2 lstm 27 columns partial indus 50%
# checkpoint_names = ['506', '538', '565', '578', '584', '592', '595', '597', '598', '599']  # 2 lstm 27 columns partial indus 75%

# checkpoint_names = ['505', '510', '515', '519', '523', '531', '533', '537', '545', '553']  # 2 gru 27 columns partial indus 5%
# checkpoint_names = ['507', '513', '516', '521', '530', '534', '539', '546', '549', '554']  # 2 gru 27 columns partial indus 10%
# checkpoint_names = ['508', '529', '532', '541', '552', '555', '561', '568', '571', '575']  # 2 gru 27 columns partial indus 25%
# checkpoint_names = ['509', '526', '551', '559', '567', '573', '579', '583', '587', '591']  # 2 gru 27 columns partial indus 50%
# checkpoint_names = ['511', '525', '543', '556', '564', '572', '576', '582', '586', '589']  # 2 gru 27 columns partial indus 75%


# checkpoint_names = ['606', '615', '632', '650', '683', '701', '720', '739', '757', '773']  # 2 lstm 33 columns partial indus 5%
# checkpoint_names = ['607', '622', '641', '655', '677', '693', '706', '719', '740', '753']  # 2 lstm 33 columns partial indus 10%
# checkpoint_names = ['609', '625', '662', '685', '711', '746', '764', '775', '796', '805']  # 2 lstm 33 columns partial indus 25%
# checkpoint_names = ['611', '630', '651', '694', '718', '751', '782', '795', '808', '816']  # 2 lstm 33 columns partial indus 50%
# checkpoint_names = ['612', '638', '687', '722', '758', '781', '801', '811', '820', '825']  # 2 lstm 33 columns partial indus 75%

# checkpoint_names = ['600', '605', '613', '634', '644', '659', '673', '691', '703', '721'] # 2 gru 33 columns partial indus 5%
# checkpoint_names = ['601', '608', '614', '626', '652', '671', '688', '710', '723', '752']  # 2 gru 33 columns partial indus 10%
# checkpoint_names = ['602', '610', '621', '637', '648', '681', '709', '733', '747', '759']  # 2 gru 33 columns partial indus 25%
# checkpoint_names = ['603', '616', '656', '692', '715', '741', '766', '786', '798', '815']  # 2 gru 33 columns partial indus 50%
# checkpoint_names = ['604', '624', '666', '702', '760', '783', '810', '818', '821', '829']  # 2 gru 33 columns partial indus 75%

# checkpoint_names = ['617', '629', '643', '665', '689', '704', '725', '750', '765', '778']  # 4 lstm 33 columns partial indus 5%
# checkpoint_names = ['618', '628', '642', '668', '679', '700', '714', '748', '772', '787']  # 4 lstm 33 columns partial indus 10%
# checkpoint_names = ['619', '674', '870', '874', '877', '883', '885', '888', '891', '895']  # 4 lstm 33 columns partial indus 25%
# checkpoint_names = ['620', '734', '803', '817', '833', '844', '852', '863', '865', '876']  # 4 lstm 33 columns partial indus 50%
# checkpoint_names = ['686', '761', '806', '830', '840', '855', '864', '871', '893', '898']  # 4 lstm 33 columns partial indus 75%

# checkpoint_names = ['627', '636', '663', '682', '697', '713', '742', '763', '785', '797']  # 4 gru 33 columns partial indus 5%
# checkpoint_names = ['631', '640', '661', '696', '732', '749', '767', '780', '794', '804']  # 4 gru 33 columns partial indus 10%
# checkpoint_names = ['633', '676', '724', '762', '807', '819', '827', '835', '843', '850']  # 4 gru 33 columns partial indus 25%
# checkpoint_names = ['675', '728', '779', '812', '831', '837', '846', '853', '856', '862']  # 4 gru 33 columns partial indus 50%
# checkpoint_names = ['635', '729', '799', '826', '841', '851', '860', '866', '872', '879']  # 4 gru 33 columns partial indus 75%

# checkpoint_names = ['645', '657', '669', '680', '690', '698', '708', '716', '745', '770']  # 5 lstm 33 columns partial indus 5%
# checkpoint_names = ['646', '678', '705', '717', '744', '769', '776', '788', '802', '813']  # 5 lstm 33 columns partial indus 10%
# checkpoint_names = ['647', '712', '735', '754', '802', '814', '823', '832', '836', '842']  # 5 lstm 33 columns partial indus 25%
# checkpoint_names = ['649', '731', '790', '834', '847', '858', '869', '880', '889', '899']  # 5 lstm 33 columns partial indus 50%
# checkpoint_names = ['653', '736', '791', '828', '848', '859', '882', '896', '900', '901']  # 5 lstm 33 columns partial indus 75%

# checkpoint_names = ['654', '672', '695', '707', '727', '738', '755', '774', '784', '789']  # 5 gru 33 columns partial indus 5%
# checkpoint_names = ['658', '684', '699', '726', '743', '756', '768', '777', '792', '800']  # 5 gru 33 columns partial indus 10%
# checkpoint_names = ['660', '873', '875', '878', '884', '886', '890', '892', '894', '897']  # 5 gru 33 columns partial indus 25%
# checkpoint_names = ['664', '730', '771', '809', '824', '838', '845', '854', '861', '868']  # 5 gru 33 columns partial indus 50%
# checkpoint_names = ['670', '737', '793', '822', '839', '849', '857', '867', '881', '887']  # 5 gru 33 columns partial indus 75%

# checkpoint_names = ['912', '916', '928', '934', '941', '946', '953', '957', '965', '973']  # 7 lstm 33 columns partial indus 5%
# checkpoint_names = ['913', '926', '936', '947', '958', '968', '978', '985', '988', '992']  # 7 lstm 33 columns partial indus 10%
# checkpoint_names = ['914', '942', '961', '979', '990', '996', '999', '1001', '1003', '1005']  # 7 lstm 33 columns partial indus 25%
# checkpoint_names = ['915', '962', '993', '1004', '1012', '1017', '1022', '1026', '1031', '1035']  # 7 lstm 33 columns partial indus 50%
# checkpoint_names = ['918', '1006', '1016', '1024', '1029', '1036', '1040', '1043', '1048', '1050']  # 7 lstm 33 columns partial indus 75%

# checkpoint_names = ['902', '917', '927', '932', '939', '945', '963', '969', '975', '980']  # 6 lstm 33 columns partial indus 5%
# checkpoint_names = ['903', '909', '919', '931', '940', '949', '959', '976', '984', '987']  # 6 lstm 33 columns partial indus 10%
# checkpoint_names = ['904', '925', '948', '989', '1000', '1008', '1010', '1014', '1018', '1023']  # 6 lstm 33 columns partial indus 25%
# checkpoint_names = ['982', '1007', '1013', '1020', '1028', '1034', '1042', '1045', '1049']  # 6 lstm 33 columns partial indus 50%  905
# checkpoint_names = ['906', '967', '1002', '1025', '1033', '1039', '1047', '1052', '1054', '1055']  # 6 lstm 33 columns partial indus 75%

# checkpoint_names = ['921', '929', '933', '938', '943', '950', '954', '955', '964', '970']  # 8 lstm 33 columns partial indus 5%
# checkpoint_names = ['922', '930', '935', '944', '952', '956', '966', '974', '977', '983']  # 8 lstm 33 columns partial indus 10%
# checkpoint_names = ['924', '937', '951', '960', '971', '981', '986', '991', '995', '997']  # 8 lstm 33 columns partial indus 25%
# checkpoint_names = ['923', '972', '998', '1009', '1015', '1019', '1027', '1032', '1037', '1041']  # 8 lstm 33 columns partial indus 50%
# checkpoint_names = ['920', '994', '1011', '1021', '1030', '1038', '1044', '1046', '1051', '1053']  # 8 lstm 33 columns partial indus 75%
