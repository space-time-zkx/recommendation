import torch
import os
import pickle
from torch_geometric.data import Data,DataLoader,HeteroData
class DataSet:
    def __init__(self,root_dir):
        self.root_dir = root_dir
        self.user_item_data = self.read_pickle(self.root_dir)
        self.timeline_data = []
        self.item_dict = {}
        item_ctr = 0
        self.train_set = []
        self.valid_set = []
        for user_id, item_data in enumerate(self.user_item_data):
            temp = []
            for time, item_id, score, review in item_data:
                if item_id not in self.item_dict:
                    self.item_dict[item_id] = item_ctr
                    item_id = item_ctr
                    item_ctr+=1
                else:
                    item_id = self.item_dict[item_id]
                temp.append([time.timestamp(),item_id,score,review])
            temp.sort(key=lambda x:x[0])
            if len(temp)>2:
                self.timeline_data.append(temp)
                self.train_set.append(temp[:-1])
                self.valid_set.append(temp[-1])
        self.item_ctr = item_ctr
    def __getitem__(self, item):

        train_data = self.train_set[item]
        valid_data = self.valid_set[item]
        rela = []
        edge_type = []
        # print(train_data)
        for i in range(len(train_data)-1):
            rela.append([train_data[i][1],train_data[i+1][1]])
            edge_type.append(0)
            rela.append([train_data[i+1][1], train_data[i][1]])
            edge_type.append(1)
        print(torch.LongTensor(rela).shape)
        graph_data = Data(y=torch.LongTensor(valid_data[1]),edges = torch.LongTensor(rela).permute(1,0),edge_type=torch.LongTensor(edge_type))
        return graph_data,self.item_ctr
    def __len__(self):
        return len(self.train_set)
    def read_pickle(self,path):
        with open(path, 'rb') as file:
            return pickle.load(file=file)
if __name__ == '__main__':
    dataset = DataSet("/user_item_data.pkl")
    test = dataset[0]
    print(test[0])

