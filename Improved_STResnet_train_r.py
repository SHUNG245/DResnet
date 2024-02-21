import torch
from Improved_STResnet import *
import pandas as pd
import numpy as np
from Improved_STResnet_dataset import *
import matplotlib.pyplot as plt
from Metric import *
from sklearn.model_selection import KFold
#datatype set to double
torch.set_default_tensor_type(torch.DoubleTensor)
path = "NN/ISTDNN/"
#train device
device = torch.device("cuda")

def sample_train(epoch_num,learningrate,batchsize,ev_num,day_num,month_num,year_num,SW,random_state,device):
    #load data
    inputpath="pm+gwr_day+time+ev.csv"
    df=pd.read_csv(inputpath)

    #划分10折训练集和测试集
    ESTIMATES=[]
    OBSERVATION=[]


    train_r2_list=[]
    train_rmse_list=[]
    train_mae_list=[]
    test_r2_list=[]
    test_rmse_list=[]
    test_mae_list=[]

    max_train_r2=0
    max_train_rmse=0
    max_train_mae= 0
    max_train={}
    max_test_r2=0
    max_test_rmse = 0
    max_test_mae = 0
    max_test={}
    max_epoch_train=0
    max_epoch_test=0

    final_train_r2=0
    final_train_rmse=0
    final_train_mae = 0
    final_test_r2=0
    final_test_rmse=0
    final_test_mae = 0

    final_max_train_r2=0
    final_max_train_rmse=0
    final_max_train_mae=0
    final_max_test_r2=0
    final_max_test_rmse=0
    final_max_test_mae=0

    n=0
    kf=KFold(n_splits=10,shuffle=True,random_state=random_state) #sample cv
    for train,test in kf.split(df):
        n+=1
        train_data=df.iloc[train]
        test_data=df.iloc[test]
        #load data
        train_dataset =STDNN_Dataset(train_data)
        test_dataset = STDNN_Dataset(test_data)
        #loader
        train_loader = DataLoader(train_dataset,batch_size= batchsize,shuffle=True,num_workers=0,drop_last=False)
        test_loader = DataLoader(test_dataset,batch_size= batchsize, shuffle= True, num_workers= 0, drop_last= False)
        #create net
        STDNN = STDNnet(train_dataset.train_x_len,ev_num=ev_num,day=day_num,month=month_num,year=year_num)
        STDNN=STDNN.to(device)

        #loss function
        loss_fn = StationweightedLoss()
        loss_fn = loss_fn.to(device)

        #optimizer
        params = [p for p in STDNN.parameters() if p.requires_grad]
        print(STDNN.parameters())
        lr= learningrate
        optimizer=torch.optim.Adam(params,lr= lr)
        #train
        epoch = epoch_num
        Epoch_list=[]
        Truevalues=[]
        Estimates=[]

        for i in range(epoch+1):
            STDNN.train()
            for data in train_loader:
                factors, targets, evs, years, months, days,weight_dem,weight_station = data
                factors = factors.to(device)
                targets = targets.to(device)
                evs = evs.to(device)
                years = years.to(device)
                months = months.to(device)
                days = days.to(device)
                weight_dem = weight_dem.to(device)
                weight_station = weight_station.to(device)

                for target in targets.cpu().numpy():
                    Truevalues.append(target)
                 #output
                outputs = STDNN(factors,evs,days,months,years)

                for output in outputs.data.cpu().numpy():
                    Estimates.append(output)

                #loss compute
                loss = loss_fn(outputs,targets,weight_dem,weight_station,super_weight=SW)
                #backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # visualization
            Epoch_list.append(i)
            train_r2_list.append(R2(Estimates, Truevalues))
            train_rmse_list.append(rmse(Estimates, Truevalues))
            train_mae_list.append(mae(Estimates,Truevalues))
            if train_r2_list[-1] >= max_train_r2:
                max_train_r2 = train_r2_list[-1]
                max_train_rmse = train_rmse_list[-1]
                max_train_mae = train_mae_list[-1]
                max_epoch_train = i

            if Epoch_list[-1] % 10 == 0:
                print("Epoch:{}, train R2:{},RMSE:{}, MAE:{}".format(Epoch_list[-1], train_r2_list[-1], train_rmse_list[-1],train_mae_list[-1]))
            Truevalues.clear()
            Estimates.clear()

            #test
            STDNN.eval()
            with torch.no_grad():
                for data in test_loader:
                    factors, targets, evs, years, months, days,weight_dem,weight_station = data
                    factors = factors.to(device)
                    targets = targets.to(device)
                    evs = evs.to(device)
                    years = years.to(device)
                    months = months.to(device)
                    days = days.to(device)

                    for target in targets.cpu().numpy():
                        Truevalues.append(target)
                        # output
                    outputs = STDNN(factors, evs, days, months, years)
                    for output in outputs.data.cpu().numpy():
                        Estimates.append(output)

                test_r2_list.append(R2(Estimates, Truevalues))
                test_rmse_list.append(rmse(Estimates, Truevalues))
                test_mae_list.append(mae(Estimates,Truevalues))

                if test_r2_list[-1] >= max_test_r2:
                    max_test_r2 = test_r2_list[-1]
                    max_test_rmse = test_rmse_list[-1]
                    max_test_mae = test_mae_list[-1]
                    max_epoch_test = i
                    if i == 1000 :
                        for e, o in zip(Estimates,Truevalues):
                            ESTIMATES.append(e)
                            OBSERVATION.append(o)

                        #torch.save(STDNN,path+"/ISTDNN"+str(n)+"_sample.pt")

                if Epoch_list[-1] % 10 == 0:
                    print("epoch: {}, test R2:{},RMSE:{}, MAE:{}".format(Epoch_list[-1], test_r2_list[-1], test_rmse_list[-1],test_mae_list[-1]))
                    #print(Estimates[0:5])
                    #print(Truevalues[0:5])
                Truevalues.clear()
                Estimates.clear()

        max_train["{}:epoch{}:train_max_r2".format(n, max_epoch_train)] = max_train_r2
        max_train["{}:epoch{}:train_max_rmse".format(n, max_epoch_train)] = max_train_rmse
        max_train["{}:epoch{}:train_max_mae".format(n, max_epoch_train)] = max_train_mae
        max_test["{}:epoch{}:test_max_r2".format(n, max_epoch_test)] = max_test_r2
        max_test["{}:epoch{}:test_max_rmse".format(n, max_epoch_test)] = max_test_rmse
        max_test["{}:epoch{}:test_max_mae".format(n, max_epoch_test)] = max_test_mae

        final_max_train_r2 += max_train_r2
        final_max_train_rmse += max_train_rmse
        final_max_train_mae += max_train_mae
        final_max_test_r2 += max_test_r2
        final_max_test_rmse += max_test_rmse
        final_max_test_mae += max_test_mae

        max_train_r2 = 0
        max_train_rmse = 0
        max_train_mae = 0
        max_test_r2 = 0
        max_test_rmse = 0
        max_test_mae = 0

        final_train_r2 = final_train_r2 + train_r2_list[-1]
        final_train_rmse = final_train_rmse + train_rmse_list[-1]
        final_train_mae = final_train_mae + train_mae_list[-1]
        final_test_r2 = final_test_r2 + test_r2_list[-1]
        final_test_rmse = final_test_rmse + test_rmse_list[-1]
        final_test_mae = final_test_mae + test_mae_list[-1]

    final_train_r2 = final_train_r2 / 10
    final_train_rmse = final_train_rmse / 10
    final_train_mae = final_train_mae / 10
    final_test_r2 = final_test_r2 / 10
    final_test_rmse = final_test_rmse / 10
    final_test_mae = final_test_mae / 10

    final_max_train_r2 = final_max_train_r2 / 10
    final_max_train_rmse = final_max_train_rmse / 10
    final_max_train_mae = final_max_train_mae / 10
    final_max_test_r2 = final_max_test_r2 / 10
    final_max_test_rmse = final_max_test_rmse / 10
    final_max_test_mae = final_max_test_mae / 10

    print("10 folds train R2:{},    RMSE:{},    MAE:{}".format(final_train_r2, final_train_rmse, final_train_mae))
    print("10 folds CV R2:{},    RMSE:{},     MAE:{}".format(final_test_r2, final_test_rmse, final_test_mae))

    print("10 folds max_train R2:{},    RMSE:{},     MAE:{}".format(final_max_train_r2, final_max_train_rmse, final_max_train_mae))
    print("10 folds max_CV R2:{},    RMSE:{},     MAE:{}".format(final_max_test_r2, final_max_test_rmse, final_max_test_mae))
    df_e=pd.DataFrame.from_dict(ESTIMATES,orient='index',columns=['estimate'])
    df_o=pd.DataFrame.from_dict(OBSERVATION,orient='index',columns=['observation'])
    df = pd.concat([df_e,df_o],axis=1)
    df.to_csv(path+"/random_{}/WDEM-345/Station_SW={}.csv".format(random_state,SW),index_label='code')
    for k, v in max_train.items():
        print("{}:  {}".format(k, v))

    for k, v in max_test.items():
        print("{}:  {}".format(k, v))

def station_train(epoch_num,learningrate,batchsize,ev_num,day_num,month_num,year_num,SW,random_state,device):
    # load data
    inputpath = "pm+gwr_day+time+ev.csv"
    df = pd.read_csv(inputpath)
    outputpath="/home/suheng/文档/深度学习/实验结果/ISTResnet_station"
    # 划分10折训练集和测试集
    Station_path = "Stations.csv"
    df_s = pd.read_csv(Station_path)
    #保存站点划分记录
    station_list_10fold=[]

    train_r2_list = []
    train_rmse_list = []
    train_mae_list = []
    test_r2_list = []
    test_rmse_list = []
    test_mae_list = []

    max_train_r2 = 0
    max_train_rmse = 0
    max_train_mae = 0
    max_train = {}
    max_test_r2 = 0
    max_test_rmse = 0
    max_test_mae = 0
    max_test = {}
    max_epoch_train = 0
    max_epoch_test = 0

    final_train_r2 = 0
    final_train_rmse = 0
    final_train_mae = 0
    final_test_r2 = 0
    final_test_rmse = 0
    final_test_mae = 0

    final_max_train_r2 = 0
    final_max_train_rmse = 0
    final_max_train_mae = 0
    final_max_test_r2 = 0
    final_max_test_rmse = 0
    final_max_test_mae = 0

    final_station_R2={}
    final_station_RMSE={}
    final_station_MAE={}
    n = 0
    kf = KFold(n_splits=10, shuffle=True, random_state=random_state)  # sample cv
    for train, test in kf.split(df_s):
        n += 1
        train_station=df_s.iloc[train]
        test_station=df_s.iloc[test]
        #把每折使用的站点列表记录下来
        station_list_10fold.append(test_station['code'].tolist())
        # 保存各站点的R2
        station_R2 = {}
        station_RMSE = {}
        station_MAE = {}
        max_station_R2 = {}
        max_station_RMSE = {}
        max_station_MAE = {}

        test_stations=[]
        test_datasets=[]
        test_dataloaders=[]
        for code in test_station['code'].tolist():
            test_stations.append(df.loc[df['code']==code])

        train_data = df.loc[df['code'].isin(train_station['code'].tolist())]
        #test_data = df.loc[df['code'].isin(test_station['code'].tolist())]

        # load data
        train_dataset = STDNN_Dataset(train_data)
        #test_dataset = STDNN_Dataset(test_data)
        for test_station_data in test_stations:
            test_datasets.append(STDNN_Dataset(test_station_data))

        # loader
        train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=0, drop_last=False)
        for test_dataset in test_datasets:
            test_dataloaders.append(DataLoader(test_dataset, batch_size=batchsize, shuffle=True, num_workers=0, drop_last=False))
        #test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True, num_workers=0, drop_last=False)

        # create net
        STDNN = STDNnet(train_dataset.train_x_len, ev_num=ev_num, day=day_num, month=month_num, year=year_num)
        STDNN = STDNN.to(device)

        # loss function
        loss_fn = StationweightedLoss()
        loss_fn = loss_fn.to(device)

        # optimizer
        params = [p for p in STDNN.parameters() if p.requires_grad]
        print(STDNN.parameters())
        lr = learningrate
        optimizer = torch.optim.Adam(params, lr=lr)
        # train
        epoch = epoch_num
        Epoch_list = []
        Truevalues = []
        Estimates = []

        for i in range(epoch + 1):
            STDNN.train()
            for data in train_loader:
                factors, targets, evs, years, months, days,weight_dem,weight_station = data
                factors = factors.to(device)
                targets = targets.to(device)
                evs = evs.to(device)
                years = years.to(device)
                months = months.to(device)
                days = days.to(device)
                weight_dem = weight_dem.to(device)
                weight_station = weight_station.to(device)

                for target in targets.cpu().numpy():
                    Truevalues.append(target)
                # output
                outputs = STDNN(factors, evs, days, months, years)

                for output in outputs.data.cpu().numpy():
                    Estimates.append(output)

                # loss compute
                loss = loss_fn(outputs,targets,weight_dem,weight_station,super_weight=SW)

                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # visualization
            Epoch_list.append(i)
            train_r2_list.append(R2(Estimates, Truevalues))
            train_rmse_list.append(rmse(Estimates, Truevalues))
            train_mae_list.append(mae(Estimates, Truevalues))
            if train_r2_list[-1] >= max_train_r2:
                max_train_r2 = train_r2_list[-1]
                max_train_rmse = train_rmse_list[-1]
                max_train_mae = train_mae_list[-1]
                max_epoch_train = i

            if Epoch_list[-1] % 10 == 0:
                print("Epoch:{}, train R2:{},RMSE:{}, MAE:{}".format(Epoch_list[-1], train_r2_list[-1],
                                                                     train_rmse_list[-1], train_mae_list[-1]))
            Truevalues.clear()
            Estimates.clear()

            # test
            STDNN.eval()
            with torch.no_grad():
                #把每个站点验证一遍验证
                testloder_num=0#第几个验证的站点
                for test_loader in test_dataloaders:
                    station_targets=[]
                    station_estimates=[]
                    for data in test_loader:
                        factors, targets, evs, years, months, days,weight_dem,weight_station = data
                        factors = factors.to(device)
                        targets = targets.to(device)
                        evs = evs.to(device)
                        years = years.to(device)
                        months = months.to(device)
                        days = days.to(device)

                        for target in targets.cpu().numpy():
                            Truevalues.append(target)
                            if i >= 800:
                                station_targets.append(target)
                            # output
                        outputs = STDNN(factors, evs, days, months, years)
                        for output in outputs.data.cpu().numpy():
                            Estimates.append(output)
                            if i >= 800:
                                station_estimates.append(output)

                    if i >= 800:
                        station_R2[station_list_10fold[-1][testloder_num]]=R2(station_estimates,station_targets)
                        station_RMSE[station_list_10fold[-1][testloder_num]]=rmse(station_estimates,station_targets)
                        station_MAE[station_list_10fold[-1][testloder_num]]=mae(station_estimates,station_targets)
                        testloder_num+=1

                test_r2_list.append(R2(Estimates, Truevalues))
                test_rmse_list.append(rmse(Estimates, Truevalues))
                test_mae_list.append(mae(Estimates, Truevalues))
                if i >=800:
                    temp_r2 = test_r2_list[-1]
                    if temp_r2 >= max_test_r2:
                        max_test_r2 = test_r2_list[-1]
                        max_test_rmse = test_rmse_list[-1]
                        max_test_mae = test_mae_list[-1]
                        max_epoch_test = i
                        max_station_R2=station_R2
                        max_station_RMSE=station_RMSE
                        max_station_MAE=station_MAE
                        torch.save(STDNN, path + "ISTDNN" + str(SW) + "_sample"+ str(n) +".pt")


                if Epoch_list[-1] % 10 == 0:
                    print("epoch: {}, test R2:{},RMSE:{}，  MAE：{}".format(Epoch_list[-1], test_r2_list[-1], test_rmse_list[-1],test_mae_list[-1]))

                Truevalues.clear()
                Estimates.clear()

        max_train["{}:epoch{}:train_max_r2".format(n, max_epoch_train)] = max_train_r2
        max_train["{}:epoch{}:train_max_rmse".format(n, max_epoch_train)] = max_train_rmse
        max_train["{}:epoch{}:train_max_mae".format(n, max_epoch_train)] = max_train_mae
        max_test["{}:epoch{}:test_max_r2".format(n, max_epoch_test)] = max_test_r2
        max_test["{}:epoch{}:test_max_rmse".format(n, max_epoch_test)] = max_test_rmse
        max_test["{}:epoch{}:test_max_mae".format(n, max_epoch_test)] = max_test_mae

        final_station_R2 = {**final_station_R2 , **max_station_R2}
        final_station_RMSE = {**final_station_RMSE ,**max_station_RMSE}
        final_station_MAE = {**final_station_MAE , **max_station_MAE}
        final_max_train_r2 += max_train_r2
        final_max_train_rmse += max_train_rmse
        final_max_train_mae += max_train_mae
        final_max_test_r2 += max_test_r2
        final_max_test_rmse += max_test_rmse
        final_max_test_mae += max_test_mae

        max_train_r2 = 0
        max_train_rmse = 0
        max_train_mae = 0
        max_test_r2 = 0
        max_test_rmse = 0
        max_test_mae = 0

        final_train_r2 = final_train_r2 + train_r2_list[-1]
        final_train_rmse = final_train_rmse + train_rmse_list[-1]
        final_train_mae = final_train_mae + train_mae_list[-1]
        final_test_r2 = final_test_r2 + test_r2_list[-1]
        final_test_rmse = final_test_rmse + test_rmse_list[-1]
        final_test_mae = final_test_mae + test_mae_list[-1]

    final_train_r2 = final_train_r2 / 10
    final_train_rmse = final_train_rmse / 10
    final_train_mae = final_train_mae / 10
    final_test_r2 = final_test_r2 / 10
    final_test_rmse = final_test_rmse / 10
    final_test_mae = final_test_mae / 10

    final_max_train_r2 = final_max_train_r2 / 10
    final_max_train_rmse = final_max_train_rmse / 10
    final_max_train_mae = final_max_train_mae / 10
    final_max_test_r2 = final_max_test_r2 / 10
    final_max_test_rmse = final_max_test_rmse / 10
    final_max_test_mae = final_max_test_mae / 10

    print("10 folds train R2:{},    RMSE:{},    MAE:{}".format(final_train_r2, final_train_rmse, final_train_mae))
    print("10 folds CV R2:{},    RMSE:{},     MAE:{}".format(final_test_r2, final_test_rmse, final_test_mae))

    print("10 folds max_train R2:{},    RMSE:{},     MAE:{}".format(final_max_train_r2, final_max_train_rmse,
                                                                    final_max_train_mae))
    print("10 folds max_CV R2:{},    RMSE:{},     MAE:{}".format(final_max_test_r2, final_max_test_rmse,
                                                                 final_max_test_mae))
    for k, v in final_station_R2.items():
        print("{}:  {}".format(k, v))
    df_r2=pd.DataFrame.from_dict(final_station_R2,orient='index',columns=['r2'])
    df_rmse=pd.DataFrame.from_dict(final_station_RMSE,orient='index',columns=['rmse'])
    df_mae=pd.DataFrame.from_dict(final_station_MAE,orient='index',columns=['mae'])
    df = pd.concat([df_r2,df_rmse,df_mae],axis=1)
    df.to_csv(outputpath+"/random_{}/WDEM-345/Station_SW={}.csv".format(random_state,SW),index_label='code')
    print(station_list_10fold)

if __name__ == '__main__':
    SW=4
    print(SW)
    #sample_train(1000,0.005,640,5,15,6,2,SW,random_state=345,device=device)
    station_train(1000,0.005,640,5,15,6,2,SW,random_state=345,device=device)


