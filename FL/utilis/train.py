import torch
import torch.nn.functional as F


def train(model, train_loader, optimizer):
    """
    本地训练
    """
    model.train()
    train_loss = 0.0
    train_acc= 0.0
    i = 0

    for id, (data, target) in enumerate(train_loader):
        # if id==0:
        #     print("测试集：",data[0]) #这边同样DPSGD的验证集也是浮点型的
        optimizer.zero_grad()  # 梯度清空
        output = model(data.to(torch.float32))  # 计算输出
        loss = F.cross_entropy(output, target.to(torch.long))  # 损失函数
        loss.backward()  # 梯度求导
        optimizer.step()  # 参数优化更新

    return train_loss,train_acc
