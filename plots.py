import json
import ast
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set()
# ax = plt.gca()
# ax.set_ylim([0.0, 1.0])


# plt(figsize=(10, 20), dpi=100)
# fig, ax = plt.subplots(2, 2)
# fig.suptitle("Learning curves with reverse transfer learning for position columns")
# # fig.set_size_inches(8.27, 11.69)
# # fig.set_size_inches(9, 4.5)
# fig.set_size_inches(9, 8)
# fig.set_dpi(100)


sns.set_style("whitegrid")
# sns.set(style='whitegrid')
# sns.color_palette("magma")
# sns.set_palette("husl")
# ax.grid(linestyle='-', linewidth=2)

def plot_learning_curves(results: list, label='', color='b'):
    # batches = [32, 64, 128, 256, 512, 1024]
    # batches = [0.0001, 0.001, 0.005, 0.01]
    # batches = ['LSTM with 2 layers', 'GRU with 2 layers', 'LSTM with 4 layers', 'GRU with 4 layers',
    #            'LSTM with 5 layers', 'GRU with 5 layers', 'LSTM with 6 layers', 'GRU with 6 layers']
    batches = ['5 layers', '6 layers', '7 layers', '8 layers']
    row = 0
    for k, r in enumerate(results):
        df = pd.DataFrame(r).transpose()
        df['mean'] = df.mean(axis=1)
        df['min'] = df.min(axis=1)
        df['max'] = df.max(axis=1)
        df['sd'] = df.std(axis=1)
        df['mean-sd'] = df['mean'] - df['sd']
        df['mean+sd'] = df['mean'] + df['sd']
        # df_invert = pd.DataFrame.from_dict(fine_tuned_results, orient="index")
        # df_invert.mean(axis=0)
        # df_invert.append(df_invert.mean(axis=0), ignore_index=True)
        # ax[k].set_style('whitegrid')
        col = k % 2
        sns.set_style("whitegrid")
        ax[row, col].set_ylim([0.0, 1.1])
        ax[row, col].set_xlim([0, 6])
        ax[row, col].plot(df.index.values.tolist(), df['mean'], color, label=label if k == 0 else "")
        # sns.lineplot(data=df, x=df.index, y='mean', color=color, label=label if k == 0 else "", palette='magma')
        # for x, y in zip(df.index.values.tolist(), df['mean']):
        #     plt.annotate(xy=(x, y), ha='center')
        # plt.plot(df.index.values.tolist(), df['mean-sd'], color + '--')
        # plt.plot(df.index.values.tolist(), df['mean+sd'], color + '--')
        ax[row, col].fill_between(df.index.values.tolist(), y1=df['mean-sd'], y2=df['mean+sd'], color=color, alpha=0.175)
        # ax[k].xlabel("portion of training samples used")
        # plt.xlabel("portion of training samples used")
        # ax[k].ylabel("test accuracy")
        # plt.ylabel("test accuracy")
        # plt.title("LR 0.01")
        ax[row, col].set_title(f"{batches[k]}")
        # ax[k].legend(loc=2)
        # plt.legend()
        row = row + 1 if col == 1 else row

    fig.legend()

    for axs in ax.flat:
        axs.set(xlabel="portion of training samples used", ylabel="test accuracy")

    for axs in ax.flat:
        axs.label_outer()


def plot_learning_curve(results: list, label='', color='b'):
    df = pd.DataFrame(results[0]).transpose()
    df['mean'] = df.mean(axis=1)
    df['min'] = df.min(axis=1)
    df['max'] = df.max(axis=1)
    df['sd'] = df.std(axis=1)
    df['mean-sd'] = df['mean'] - df['sd']
    df['mean+sd'] = df['mean'] + df['sd']
    # df_invert = pd.DataFrame.from_dict(fine_tuned_results, orient="index")
    # df_invert.mean(axis=0)
    # df_invert.append(df_invert.mean(axis=0), ignore_index=True)
    plt.ylim([0.0, 1.0])
    plt.xlim([0, 9])
    plt.plot(df.index.values.tolist(), df['mean'], label=label)
    # sns.lineplot(data=df, x=df.index, y='mean', color=color, label=label if k == 0 else "", palette='magma')
    # for x, y in zip(df.index.values.tolist(), df['mean']):
    #     plt.annotate(xy=(x, y), ha='center')
    # plt.plot(df.index.values.tolist(), df['mean-sd'], color + '--')
    # plt.plot(df.index.values.tolist(), df['mean+sd'], color + '--')
    plt.fill_between(df.index.values.tolist(), y1=df['mean-sd'], y2=df['mean+sd'], alpha=0.175)
    # ax[k].xlabel("portion of training samples used")
    plt.xlabel("portion of training samples used")
    # ax[k].ylabel("test accuracy")
    plt.ylabel("test accuracy")
    plt.title("Learning curves with 5-64 LSTM")
    # ax[k].set_title(f"learning rate {batches[k]}")
    # ax[k].legend(loc=2)
    plt.legend(loc=2)


def compare_learning_itself(results, title=''):
    df = pd.DataFrame(results).transpose()
    df.rename(columns={0: '1_128', 1: '2_64', 2: '4_32', 3: '2_128', 4: '5_128', 5: '3_256'}, inplace=True)
    df.plot()
    plt.xlabel("training samples")
    plt.ylabel("test accuracy")
    plt.title("Learning curve for 42 seed for " + title)


def compare_learning_for_exp(results, title=''):
    df = pd.DataFrame(results).transpose()
    df.rename(columns={0: 'fine_tuning', 1: 'training'},
              inplace=True)  # , 2: 'fine_tuned_only_last_layer', 3: 'bi-di fine tuning', 4: 'bi-di training'
    df.plot()
    plt.xlabel("training samples")
    plt.ylabel("test accuracy")
    plt.title("Learning curve for 42 seed for " + title)


def compare_partial_industry_pre_training(results, color='b'):
    df = pd.DataFrame(results).transpose()
    df['mean'] = df.mean(axis=1)
    df['sd'] = df.std(axis=1)
    df['mean-sd'] = df['mean'] - df['sd']
    df['mean+sd'] = df['mean'] + df['sd']
    # plt.plot(df.index.values.tolist(), df['mean'])
    # plt.plot(df.index.values.tolist(), df['mean-sd'], '--')
    # plt.plot(df.index.values.tolist(), df['mean+sd'], '--')
    # plt.fill_between(df.index.values.tolist(), y1=df['mean-sd'], y2=df['mean+sd'], alpha=0.3)
    #
    # plt.xlabel("portion of training samples of industry used for pre-training")
    # plt.ylabel("test accuracy of phume fine tuning")
    # plt.title("Learning curve lstm 8-64 33 columns partial industry")

    ax = sns.lineplot(x=df.index.values.tolist(), y='mean', data=df, color=color)
    ax.set_ylim([0.8, 0.9])
    ax.set_xlim([0, 6])
    plt.fill_between(df.index.values.tolist(), y1=df['mean-sd'], y2=df['mean+sd'], color=color, alpha=0.175)
    ax.set(xlabel='portion of training samples of industry dataset used for pre-training',
           ylabel='test accuracy',
           title='Learning curve of phume data fine tuned with LSTM 5 layers \n for sensor acceleration columns')
    # plt.legend(loc=0)


####################################################
files = ['result', 'result2', 'result3']
for file in files:
    with open(f'phume/{file}.txt', "r") as f:
        fine_tuned_results = f.read()

    results = []
    for x in fine_tuned_results.split('\n'):
        d: dict = ast.literal_eval(x)
        portions = []
        for _, v in d.items():
            new_d = {}
            for k, v2 in v.items():
                new_d[k] = v2[0] if isinstance(v2, list) else v2
            portions.append(new_d)
        results.append(portions)

    if file == 'result':
        label = 'phume all layers fine tuning'
    elif file == 'result2':
        label = 'phume last 3 layers fine tuning'
    else:
        label = 'phume last classifier layer fine tuning'

# results = {}
# i = 0
# partial_usage = ['0%', '5%', '10%', '25%', '50%', '75%', '100%']
# for x in fine_tuned_results.split('\n'):
#     d: dict = ast.literal_eval(x)
#     results[partial_usage[i]] = []
#     for _, v in d.items():
#         results[partial_usage[i]].append(v['1.0'])
#     i = i + 1

# results = {}
# i = 0
# partial_usage = ['0%', '5%', '10%', '25%', '50%', '75%', '100%']
# for x in fine_tuned_results.split('\n'):
#     d: dict = ast.literal_eval(x)
#     results[partial_usage[i]] = []
#     for _, v in d.items():
#         results[partial_usage[i]].append(v)
#     i = i + 1
# df = []
# for k, v in results:
#     df = pd.DataFrame(v)

    plot_learning_curve(results, label=label, color='#3000F0')  # 3000F0 378805
# compare_partial_industry_pre_training(results, color='#3000F0')
# compare_learning_itself(results, 'phume fine tuning')
# compare_learning_for_exp(results, '3-256 GRU')
####################################################

with open("phume/result2_train.txt", "r") as f:
    phume_train_test = f.read()

results_train = []
for x in phume_train_test.split('\n'):
    d: dict = ast.literal_eval(x)
    portions = []
    for _, v in d.items():
        portions.append(v)
    results_train.append(portions)

plot_learning_curve(results_train, label='phume training', color='#F02C10')  # F02C10
# compare_learning_itself(results_train, 'phume training')

####################################################

plt.savefig('learning curves for 5-64 LSTM partial fine tuning.png')
plt.show()
