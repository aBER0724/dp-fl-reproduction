from torch.utils.data import DataLoader

def minibatch_loader(dataset, minibatch_size, iterations):
    """
    将数据集划分为多个 minibatch
    """
    return DataLoader(
        dataset, 
        batch_sampler=EquallySizedAndIndependentBatchSamplerWithoutReplace(dataset, minibatch_size, iterations)
    )

def microbatch_loader(microbatch_size, minibatch, drop_last=True):
    """
    将 minibatch 再细分为 microbatch
    """
    return DataLoader(
        minibatch,
        batch_size=microbatch_size,
        drop_last=drop_last # 丢弃未到达 batch_size 的数据
    )