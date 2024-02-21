import torch
import Data_load
from torch import nn
import numpy as np

class STDNnet(nn.Module):
    def __init__(self,train_x_size,ev_num,day,month,year):
        super().__init__()
        self.temporalModule=TemporalModule(day,month,year)
        self.spatialModule = SpatialModule(ev_num)
        self.factorModule = FactorModule(train_x_size)
        self.relu=nn.ReLU()
        self.bn0 = nn.BatchNorm1d(train_x_size+ev_num+day+month+year)
        self.hd_layer1 = nn.Linear(train_x_size+ev_num+day+month+year, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.hd_layer2 = nn.Linear(64,32)
        self.bn2 = nn.BatchNorm1d(32)
        self.hd_layer3 = nn.Linear(32,16)
        self.bn3 = nn.BatchNorm1d(16)
        self.hd_layer4 = nn.Linear(16,32)
        self.bn4 = nn.BatchNorm1d(32)
        self.hd_layer5 = nn.Linear(32,64)
        self.bn5 = nn.BatchNorm1d(64)
        self.hd_layer6 = nn.Linear(64,train_x_size+ev_num+day+month+year)
        self.bn6 = nn.BatchNorm1d(train_x_size+ev_num+day+month+year)
        self.hd_layer7 = nn.Linear(train_x_size+ev_num+day+month+year,1)

    def tensor_merge(self,x,ev,day,month,year):
        x = torch.cat([x, ev,day,month,year], 1)
        return x

    def forward(self,x,evs,day,month,year):
        x = self.factorModule(x)
        ev = self.spatialModule(evs)
        day,month,year = self.temporalModule(day,month,year)
        merge_data = self.tensor_merge(x,ev,day,month,year)

        input= self.bn0(merge_data)
        out1 = self.hd_layer1(input)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)

        out2 = self.hd_layer2(out1)
        out2 = self.bn2(out2)
        out2 = self.relu(out2)

        out3 = self.hd_layer3(out2)
        out3 = self.bn3(out3)
        out3 = self.relu(out3)

        out4 = self.hd_layer4(out3)
        out4 = self.bn4(out4)
        out4 = self.relu(out4)

        out5 = self.hd_layer5(out4+out2)
        out5 = self.bn5(out5)
        out5 = self.relu(out5)

        out6 = self.hd_layer6(out5+out1)
        out6 = self.bn6(out6)
        out6 = self.relu(out6)

        output = self.hd_layer7(out6+input)
        return output.squeeze(-1)




class SpatialModule(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.ev_filter = nn.Linear(8, m, bias=False)

    def forward(self, evs):
        ev_filter = self.ev_filter(evs)
        return ev_filter

class TemporalModule(nn.Module):
    def __init__(self,day_len,month_len,year_len):
        super().__init__()
        self.day_embedding=nn.Linear(31,day_len,bias=False)
        self.month_embedding=nn.Linear(12,month_len,bias=False)
        self.year_embedding=nn.Linear(4,year_len,bias=False)
    def forward(self,day,month,year):
        day_matrix=self.day_embedding(day)
        month_matrix=self.month_embedding(month)
        year_matrix=self.year_embedding(year)
        return day_matrix,month_matrix,year_matrix

class FactorModule(nn.Module):
    def __init__(self,train_x_size):
        super().__init__()
        self.attention=nn.Sequential(
            nn.Linear(train_x_size,train_x_size,bias=False),
            nn.Sigmoid()
        )


    def forward(self,x):
        Impotance_score = self.attention(x)
        x = torch.mul(x, Impotance_score)
        return x

class StationweightedLoss(nn.Module):
    def __int__(self):
        super(StationweightedLoss, self).__int__()

    def forward(self,x,y,dem_weight,super_weight):
        error_pow = torch.pow((x - y),2)
        weight_dem = torch.exp(dem_weight)
        #weight_station = torch.exp(-station_weight/10)
        #return super_weight * torch.mean(error_pow*weight_dem*weight_station)
        return super_weight * torch.mean(error_pow*weight_dem)

