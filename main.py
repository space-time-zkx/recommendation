from torch_geometric.data import DataLoader
import torch
import torch.optim as optim
from model import RecGraph
from dataloader.dataset import DataSet
from torch.autograd import Variable
from torch.cuda.amp import autocast, GradScaler
import torch.nn as nn
def trainBatch(train_iter, net, criterion, optimizer):
    
    data = train_iter.next()
    with autocast():
        preds = net(data[0].cuda())
    
    # print(preds.shape,data[0].y.shape,data[0].y)
    cost = criterion(preds, data[0].y.long().cuda())
    # cost.backward()
    scaler.scale(cost).backward()
    # optimizer.step()
    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    return cost
class averager(object):
    """Compute average for `torch.Variable` and `torch.Tensor`. """

    def __init__(self):
        self.reset()

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res
if __name__ == '__main__':
    dataset = DataSet("/root/autodl-tmp/baseline-main/cache/user_item_data.pkl")
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    num_items = dataset[0][1]
    model = RecGraph(num_items).cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.001,
                           betas=(0.5, 0.999))
    criterion = torch.nn.CrossEntropyLoss()
    scaler = GradScaler()
    for epoch in range(20):
        print("epoch:",epoch)
        model.train()
        loss_avg = averager()
        train_iter = iter(train_loader)
        i = 0
        while i < len(train_loader):
            cost = trainBatch(train_iter, model, criterion, optimizer)
            print(cost)
            i += 1
            loss_avg.add(cost)
    
    torch.save(model,"net.pth")

