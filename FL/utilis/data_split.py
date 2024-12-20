import numpy as np
from test_utils.output_figue import data_split_figure


def dirichlet_split_noniid(train_class, train_labels, alpha, n_clients, seed):
    """
    参数为 alpha 的 Dirichlet 分布将数据索引划分为 n_clients 个子集\n
    狄利克雷分布相关函数
    """
    np.random.seed(seed)
    train_labels = torch.tensor(train_labels)
    n_classes = train_labels.max() + 1

    # 记录的每个class在client上分布的概率
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)

    # 记录每个类别对应的样本下标
    class_idcs = [np.argwhere(train_labels == y).flatten() for y in range(n_classes)]

    client_idcs = [[] for _ in range(n_clients)]

    # 记录N个client分别对应样本集合的索引
    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    data_split_figure(train_labels, train_class, client_idcs, n_clients, n_classes)

    """这里返回的是一个二维list，每个二级list装了对应下标的client分配到的数据的索引"""
    return client_idcs
