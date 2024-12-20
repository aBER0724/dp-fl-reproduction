import torch.utils.data
import copy
import torch.nn as nn
from datasets.get_data import get_data
from model.CNN import CNN
from FL.utilis.data_split import dirichlet_split_noniid
from FL.utilis.train import train


def send_global_model_to_clients(global_model, clients_model_list):
    """
    将全局模型发送给所有客户端
    """
    with torch.no_grad(): # 临时关闭梯度计算
        for i in range(len(clients_model_list)):
            clients_model_list[i].load_state_dict(global_model.state_dict(), strict=True)

    return clients_model_list


def local_clients_sgd(clients_model_list, clients_optimizer_list, clients_criterion_list, epoch_num):
    """
    客户端本地梯度下降
    """
    # 循环客户端
    for i in range(clients_model_list):
        # 获取模型,优化器,损失函数
        model = clients_model_list[i]
        optimizer = clients_optimizer_list[i]
        criterion = clients_criterion_list[i]

        # 计算 batch 大小(取整)
        # batch_size = math.floor(len(clients_data_list[i]) * q)
        batch_size = 256
        minibatch_size = batch_size # data -> mini_batch
        microbatch_size = 1  # mini_batch -> micro_batch
        iterations = 1  # n个batch，这边就定一个，每次训练采样一个Lot

    return


def update_global_model():
    """
    服务器集合客户端模型参数更新全局模型
    """


    return


def create_model_optimizer_criterion_dict(number_of_clients, learning_rate, model):
    """
    创建各个客户端的本地模型,优化器, 损失函数
    """
    clients_model_list = []
    clients_optimizer_list = []
    clients_criterion_list = []

    for i in range(number_of_clients):
        # model_info = CNN_tanh()
        # model_info.load_state_dict(model.state_dict())
        model_info = copy.deepcopy(model)
        clients_model_list.append(model_info)

        optimizer_info = torch.optim.SGD(model_info.parameters(), lr=learning_rate)
        clients_optimizer_list.append(optimizer_info)

        criterion_info = nn.CrossEntropyLoss()
        clients_criterion_list.append(criterion_info)

    return clients_model_list, clients_optimizer_list, clients_criterion_list


def fed_avg(train_data, test_data, model, client_num, iters, seed, batch_size ,local_epochs):

    # 标签
    train_labels = torch.tensor(train_data.targets)

    # 客户端样本分配
    clients_data_list = dirichlet_split_noniid(train_data.classes,train_labels,100,client_num,seed)

    # 各个客户端的model,optimizer,criterion的分配
    clients_model_list, clients_optimizer_list, clients_criterion_list = create_model_optimizer_criterion_dict(client_num, learning_rate,model)

    # 初始化全局模型
    global_model = model

    # 加载测试数据
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    print(f"联邦学习开始--------------------------")

    for i in range(iters):

        print(f"第{i}轮联邦学习----------------------")

        # 1. 广播全局模型
        clients_models_list = send_global_model_to_clients(global_model)

        # 2. 本地随机梯度下降
        local_clients_sgd(clients_model_list, clients_optimizer_list, clients_criterion_list, local_epochs)

        # 3. 接受上传参数并更新全局模型
        global_model = update_global_model(clients_model_list)

    return


if __name__ == "__main__":

    dataset_name = 'mnist'

    # 将数据集分为训练集和测试集
    dataset_train, dataset_test = get_data(dataset_name)

    # 设置图像大小为训练集的第一幅图像大小
    img_size = dataset_train[0][0].shape

    # 设置模型
    model = CNN()

    batch_size = 64 # 批次大小
    local_epochs = 10 # 本地梯度下降轮次
    learning_rate = 0.01 # 学习率
    client_num = 10 # 客户端数量
    seed = 1 # 随机种子

    fed_avg(dataset_train,dataset_test,model,client_num,10,seed,batch_size,local_epochs)
