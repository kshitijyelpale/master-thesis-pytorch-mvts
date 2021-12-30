import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

WINDOW_SIZE = 60


def prepare_sequences(data_frame, encoded_labels):
    sequences = []
    for i in range(data_frame.shape[0] - WINDOW_SIZE):
        sequence = data_frame.iloc[i: i + WINDOW_SIZE]
        label = encoded_labels[i + WINDOW_SIZE - 1]
        sequences.append((sequence, label))

    return sequences


def prepare_sequences_for_predictions(data_frame):
    sequences = []
    for i in range(data_frame.shape[0] - (2 * WINDOW_SIZE)):
        sequence = data_frame.iloc[i: i + WINDOW_SIZE]
        label = data_frame.iloc[i + WINDOW_SIZE: i + (2 * WINDOW_SIZE)]
        sequences.append((sequence, label))

    return sequences


def get_sequences_from_sequence_ids(dataset, sequence_id_label, sequence_ids, limited_columns=None, drop_columns=None,
                                    regression=False):
    data_frame = dataset[dataset[sequence_id_label].isin(sequence_ids)].copy()
    # print(len(data_frame[sequence_id_label].unique()), len(sequence_ids))
    en_labels = data_frame["en_label"].values.tolist()

    if drop_columns:
        data_frame.drop(columns=drop_columns, inplace=True)

    if limited_columns:
        data_frame = data_frame[limited_columns]

    if regression is True:
        return prepare_sequences_for_predictions(data_frame)

    return prepare_sequences(data_frame, en_labels)


def show_confusion_matrix(confusion_matrix):
    hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
    plt.ylabel('True motion')
    plt.xlabel('Predicted motion')


def pre_process_data(dataset, used_columns, sequence_id_label, label_column_name, partial_sequences=None,
                     regression=False):
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(dataset[label_column_name])
    dataset["en_label"] = encoded_labels
    # used_columns = [col for col in dataset.columns if col.startswith('velocity')]
    # phume.drop(['label'], axis=1, inplace=True)
    # sequence_id_label = 'sequence_id'
    sequence_ids = dataset[sequence_id_label].unique()
    if partial_sequences:
        sequence_ids = sequence_ids[0:int(len(sequence_ids) * partial_sequences)]

    # drop_columns = ['label', 'en_label', 'subject_id', 'sequence_id', 'grouped_sequence_id']

    # splitting sequence ids into train-val-test sequence ids
    train_sequence_ids, test_sequence_ids = train_test_split(sequence_ids, test_size=0.2)
    train_sequence_ids, val_sequence_ids = train_test_split(train_sequence_ids, test_size=0.125)
    print(len(train_sequence_ids), len(val_sequence_ids), len(test_sequence_ids))

    train_sequences = get_sequences_from_sequence_ids(dataset, sequence_id_label, train_sequence_ids,
                                                      limited_columns=used_columns, regression=regression)

    val_sequences = get_sequences_from_sequence_ids(dataset, sequence_id_label, val_sequence_ids,
                                                    limited_columns=used_columns, regression=regression)

    test_sequences = get_sequences_from_sequence_ids(dataset, sequence_id_label, test_sequence_ids,
                                                     limited_columns=used_columns, regression=regression)
    print(f'Data transformation done')
    print(len(train_sequences), len(val_sequences), len(test_sequences))

    # with open("tmp.txt", "a") as f:
    #     data_size = f'{len(train_sequences), len(val_sequences), len(test_sequences)}\n'
    #     f.write(data_size)

    return train_sequences, val_sequences, test_sequences, label_encoder
