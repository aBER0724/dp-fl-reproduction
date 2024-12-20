from matplotlib import pyplot as plt

def data_split_figure(train_labels, train_class, client_idcs, n_clients, n_classes):
    """
    输出客户端数据分布图
    """
    plt.figure(figsize=(12, 8))
    plt.hist([train_labels[idc] for idc in client_idcs], stacked=True,
             bins=np.arange(min(train_labels) - 0.5, max(train_labels) + 1.5, 1),
             label=["Client {}".format(i) for i in range(n_clients)],
             rwidth=0.5)
    plt.xticks(np.arange(n_classes), train_class)
    plt.xlabel("Label type")
    plt.ylabel("Number of samples")
    plt.legend(loc="upper right")
    plt.title("Display Label Distribution on Different Clients")
    plt.show()