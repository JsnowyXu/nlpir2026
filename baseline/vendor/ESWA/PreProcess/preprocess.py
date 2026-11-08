import json
import os
import re
import pandas as pd
from tqdm import tqdm
import numpy as np

from filterData import judgeFile

dirpath = os.path.dirname(os.path.realpath(__file__)) +'/'
print (dirpath)
with open(dirpath + 'config/config.json', 'r') as f:
    data_config = json.load(f)

keyF = 'new-US'
key = 'US-filter'

def filterPatentYear(raw_path=dirpath + data_config[key]['reIndex_path'], write_path=dirpath + data_config[keyF]['filter_com_year_path']):
    '''
    筛选公司：在2018,2019-2020年均有数据，2010-2017年也有数据
    从data_config[key]['reIndex_path']中读取数据,这个数据目前是重复专利数据，专利：公司：分类号=1：1：1
    存储在data_config[key]['filter_com_year_path']中：保存的是经过年份筛选后的专利数据
    '''
   
    dt = pd.read_csv(raw_path, index_col=0)
    dt.sort_values(by='publicationDate', inplace=True)
    if keyF.endswith('US'):
        dt['date_time'] = pd.to_datetime(dt['publicationDate'], format="%Y%m%d")
    else:
        dt['date_time'] = pd.to_datetime(dt['publicationDate'])
    dt['year'] = dt['date_time'].apply(lambda x: x.year)
    dt = dt[dt['year'] < 2020]
    filter_data = []
    for cid, patents in tqdm(dt.groupby(by='orgID')):
        years = patents.year.values.tolist()
        years = set(years)
        if 2018 not in years or (2019 not in years):
            continue
        res = years - set([2018, 2019])
        if len(res) == 0:
            continue
        filter_data.append(patents)
    filter_data = pd.concat(filter_data, axis=0)
    filter_data.sort_values(by='publicationDate', inplace=True)
    for year, patents in filter_data.groupby(by='year'):
        print("year is:{}, count:{}".format(year, len(patents)))
    del filter_data['date_time'], filter_data['year']
    judgeFile(write_path)
    print('filter years:' + str(len(filter_data)))
    filter_data.to_csv(write_path)

def filterPatentCount(raw_path=dirpath + data_config[keyF]['filter_com_year_path'], write_path=dirpath + data_config[keyF]['com_filt_path'],
                MAX_NUM=data_config[keyF]['MAX_NUM'], keys=data_config[keyF]['keys'],
                map_path=data_config[keyF]['map_ID']):
    '''
    筛选出公司发表专利分布在训练、验证和测试集区间都有分布的专利，且提取总数不超过20万（这里专利号是重复的，后面可以增加一下MaxNum以增加数据集数量）
    '''
    # 从公司层面出发，对每个公司筛选至多300条数据，测试集上至多有50条数据
    dt = pd.read_csv(raw_path, index_col=0)
    dt.sort_values(by='publicationDate', inplace=True)
    if keyF.endswith('US'):
        dt['date_time'] = pd.to_datetime(dt['publicationDate'], format="%Y%m%d")
    else:
        dt['date_time'] = pd.to_datetime(dt['publicationDate'])
    dt['year'] = dt['date_time'].apply(lambda x: x.year)
    del dt['date_time']
    all_datas = []
    train_years = range(2010, 2018)
    val_years = [2018]
    test_years = [2019]
    count = 0
    dict_com = {}
    patents_count = {}
    for cid, patents in tqdm(dt.groupby(by='orgID')):
        dict_com[cid] = len(patents)
    dict_com = sorted(dict_com.items(), key=lambda x: x[1], reverse=True)
    for i, (com, ct) in enumerate(tqdm(dict_com)):
        patents = dt[dt['orgID'] == com]
        historical_dt = patents[patents['year'].isin(train_years)]
        val_dt = patents[patents['year'].isin(val_years)]
        test_dt = patents[patents['year'].isin(test_years)]

        historical_dt_train = []
        historical_dt_val = []
        historical_dt_test= []
        for pid, uniquepatents in tqdm(historical_dt.groupby(by='patentId')):
            #    uniquepatents_sort = uniquepatents.sort_values(by='publicationDate')
               historical_dt_train.append(uniquepatents)
        for pid, uniquepatents in tqdm(val_dt.groupby(by='patentId')):
               historical_dt_val.append(uniquepatents)
        for pid, uniquepatents in tqdm(test_dt.groupby(by='patentId')):
               historical_dt_test.append(uniquepatents)
       
        print(len(historical_dt_train),len(historical_dt_val),len(historical_dt_test))
        train_dt_srt =historical_dt_train[-300:]
        val_dt_srt =historical_dt_val[:50]
        test_dt_srt =historical_dt_test[:50]
        com_ds = []
        com_ds.append(train_dt_srt)
        com_ds.append(val_dt_srt)
        com_ds.append(test_dt_srt)
        all_datas.append(pd.concat(train_dt_srt, ignore_index=True))
        all_datas.append(pd.concat(val_dt_srt, ignore_index=True))
        all_datas.append(pd.concat(test_dt_srt, ignore_index=True))
        # count += len(historical_dt_test[:50])
        for dflist in [item for item in com_ds]:
            for df in dflist:
                key = df['patentId'].iloc[0]
                if key not in patents_count:
                    patents_count[key] = 1
        if len(patents_count) > MAX_NUM:
            break
    all_dts = pd.concat(all_datas, axis=0)
    print('filter 300 50 50 count:' + str(len(all_dts)))
    judgeFile(write_path)
    all_dts.to_csv(write_path)
    
def process(key='US-filter',raw_path=dirpath + data_config[keyF]['com_filt_path'], ):
    '''
    将com和code合并，不重复专利号
    '''
    historical_dt = pd.read_csv(raw_path)
    keys = ["orgID", "codeId", "codeIVId", "codeIIIId", "codeIIId", "codeIId"]
    dict_dt = {}
    for k in keys:
        dict_dt[k] = []
    dict_dt["patentId"] = []# 不重复专利号
    idx = 0
    for pid, patent in tqdm(historical_dt.groupby(by='patentId')):
        idx += 1
        dict_dt['patentId'].append(pid)
        for k in keys:
            value = list(set(patent[k].values.tolist()))
            value = map(lambda x: str(x), value)
            dict_dt[k].append(";".join(value))
    dict_pd = pd.DataFrame(dict_dt)
    judgeFile(dirpath + data_config[keyF]['concate_path'])
    print('merge patents:' + str(len(dict_pd)))
    dict_pd.to_csv(dirpath + data_config[keyF]['concate_path'])

def Reindex(raw_path=dirpath + data_config[keyF]['concate_path'], write_path=dirpath + data_config[keyF]['reidx_path'],
              keys=data_config[keyF]['keys'],
                map_path=data_config[keyF]['map_ID']):
    '''
    将筛选过后的分类号重新索引
    '''
    all_dts = pd.read_csv(raw_path, index_col=0)
    for key, map_p in zip(keys, map_path):
        dict_lb = {}
        def count(x):
            for x_i in x:
                if x_i not in dict_lb:
                    dict_lb[x_i] = 0
                else:
                    dict_lb[x_i] += 1
        all_dts[key].apply(lambda x: count(x.split(";")))
        raw_ids = list(dict_lb.keys())
        new_ids = {i_d: i for i, i_d in enumerate(raw_ids)}

        rawIds, newIds = zip(*new_ids.items())
        idmaps = pd.DataFrame({key: rawIds, key + "reId": newIds})
        print(map_p + ':' + str(len(idmaps)))
        judgeFile(dirpath + map_p)
        idmaps.to_csv(dirpath + map_p)
        all_dts[key] = all_dts[key].apply(lambda x: ";".join([str(new_ids[i]) for i in x.split(";")]))
    all_dts.to_csv(write_path)
    print('reindex all:' + str(len(all_dts)))
    print('test')


def filterYear(raw_path=dirpath + data_config[keyF]['root_path'], write_path=dirpath + data_config[keyF]['filt_root_path']):
    # 筛选公司：在2018,2019-2020年均有数据，2010-2017年也有数据
    # 从data_config[key]['root_path']中读取数据
    # 存储在data_config[key]['filt_root_path']中
    dt = pd.read_csv(raw_path, index_col=0)
    dt.sort_values(by='publicationDate', inplace=True)
    if keyF.endswith('US'):
        dt['date_time'] = pd.to_datetime(dt['publicationDate'], format="%Y%m%d")
    else:
        dt['date_time'] = pd.to_datetime(dt['publicationDate'])
    dt['year'] = dt['date_time'].apply(lambda x: x.year)
    dt = dt[dt['year'] < 2020]
    filter_data = []
    for cid, patents in tqdm(dt.groupby(by='orgID')):
        years = patents.year.values.tolist()
        years = set(years)
        if 2018 not in years or (2019 not in years):
            continue
        res = years - set([2018, 2019])
        if len(res) == 0:
            continue
        filter_data.append(patents)
    filter_data = pd.concat(filter_data, axis=0)
    filter_data.sort_values(by='publicationDate', inplace=True)
    for year, patents in filter_data.groupby(by='year'):
        print("year is:{}, count:{}".format(year, len(patents)))
    del filter_data['date_time'], filter_data['year']
    judgeFile(write_path)
    filter_data.to_csv(write_path)

def filterCount(raw_path=dirpath + data_config[keyF]['filt_root_path'], write_path=dirpath + data_config[keyF]['com_filt_path'],
                MAX_NUM=data_config[keyF]['MAX_NUM'], keys=data_config[keyF]['keys'],
                map_path=data_config[keyF]['map_ID']):
    # 从公司层面出发，对每个公司筛选至多300条数据，测试集上至多有50条数据
    dt = pd.read_csv(raw_path, index_col=0)
    dt.sort_values(by='publicationDate', inplace=True)
    if keyF.endswith('US'):
        dt['date_time'] = pd.to_datetime(dt['publicationDate'], format="%Y%m%d")
    else:
        dt['date_time'] = pd.to_datetime(dt['publicationDate'])
    dt['year'] = dt['date_time'].apply(lambda x: x.year)
    del dt['date_time']
    all_datas = []
    train_years = range(2010, 2018)
    val_years = [2018]
    test_years = [2019]
    count = 0
    dict_com = {}
    for cid, patents in tqdm(dt.groupby(by='orgID')):
        dict_com[cid] = len(patents)
    dict_com = sorted(dict_com.items(), key=lambda x: x[1], reverse=True)
    for i, (com, ct) in enumerate(tqdm(dict_com)):
        patents = dt[dt['orgID'] == com]
        historical_dt = patents[patents['year'].isin(train_years)]
        val_dt = patents[patents['year'].isin(val_years)]
        test_dt = patents[patents['year'].isin(test_years)]

        historical_dt_sort = historical_dt.sort_values(by='publicationDate')
        val_dt_srt = val_dt.sort_values(by='publicationDate')
        test_dt_srt = test_dt.sort_values(by='publicationDate')

        all_datas.append(historical_dt_sort[-300:])
        all_datas.append(val_dt_srt[:50])
        all_datas.append(test_dt_srt[:50])

        count += len(historical_dt_sort[-300:])
        count += len(val_dt_srt[:50])
        count += len(test_dt_srt[:50])
        # count += len(future_dt)

        if count > MAX_NUM:
            break
    all_dts = pd.concat(all_datas, axis=0)
    for key, map_p in zip(keys, map_path):
        dict_lb = {}

        def count(x):
            for x_i in x:
                if x_i not in dict_lb:
                    dict_lb[x_i] = 0
                else:
                    dict_lb[x_i] += 1

        all_dts[key].apply(lambda x: count(x.split(";")))
        raw_ids = list(dict_lb.keys())
        new_ids = {i_d: i for i, i_d in enumerate(raw_ids)}

        rawIds, newIds = zip(*new_ids.items())
        idmaps = pd.DataFrame({key: rawIds, key + "reId": newIds})
        judgeFile(dirpath + map_p)
        idmaps.to_csv(dirpath + map_p)

        all_dts[key] = all_dts[key].apply(lambda x: ";".join([str(new_ids[i]) for i in x.split(";")]))
    judgeFile(write_path)
    all_dts.to_csv(write_path)
    
def adjacentId(raw_path=dirpath + data_config[keyF]['dt_IIId_path'], to_path=dirpath + data_config[keyF]['adj_IIId_path']):
    # 构建第i层的邻接矩阵，如果两个分类号是由同一个大类产生的，则两个分类号之间权重为1
    dt = pd.read_csv(raw_path)
    total_num = len(dt)
    columns = dt.columns
    np_dt = np.zeros([total_num, total_num])
    dt['high_level'] = dt[columns[-1]].apply(lambda x: x[:-1])
    for highId, codes in dt.groupby(by='high_level'):
        codeId = codes[columns[2]].values
        for i in codeId:
            for j in codeId:
                np_dt[i][j] = 1
    print(np_dt.nonzero())
    judgeFile(to_path)
    np.save(to_path, np_dt)

def hierarchy_tree(raw_path=data_config[keyF]["map_hierarchy_tree"], to_path=data_config[keyF]['new_tree']):
    # 构建上下层关系树
    def catchPatent(x, rule_code):
        res = rule_code.match(x).group()
        return res

    low_path =dirpath + raw_path[0]
    low_data = pd.read_csv(low_path, index_col=0)
    for i in range(1, len(raw_path)):
        high_path = dirpath +  raw_path[i]
        high_data = pd.read_csv(high_path, index_col=0)
        low_key = data_config[keyF]['Id_low'][i - 1]
        high_key = data_config[keyF]['Id_high'][i - 1]
        print(low_key, high_key)
        rule_code = re.compile(data_config[keyF]['match_rule'][i - 1])
        print(rule_code)
        low_data[high_key] = low_data[low_key].apply(lambda x: catchPatent(x, rule_code))
        print(high_data.columns, low_data.columns)
        dt = pd.merge(high_data, low_data)
        del dt[high_key], dt[low_key], dt[high_key+"Id"], dt[low_key+"Id"]
        print("save data: "+str(len(dt))+" in path {}".format(dirpath + to_path[i-1]))
        judgeFile(dirpath + to_path[i - 1])
        dt.to_csv(dirpath + to_path[i - 1], index=False)
        low_data = high_data


def MAP_rawID(text_path=data_config[keyF]['Id_path'], raw_path=data_config[keyF]['map_ID'], to_path=data_config[keyF]['map_hierarchy_tree']):
    for text, raw, map in zip(text_path, raw_path, to_path):
        raw_dt = pd.read_csv(dirpath + text)
        new_dt = pd.read_csv(dirpath + raw, index_col=0)
        renew_dt = pd.merge(new_dt, raw_dt)
        judgeFile(dirpath + map)
        print(map + ':' + str(len(map)))

        renew_dt.to_csv(dirpath + map)

def reId_all(key='US-filter'):
    reIds = pd.read_csv(dirpath + data_config[keyF]['reidx_path'], index_col=0)
    raw_file = pd.read_csv(dirpath + data_config[key]['filter_code_path'], index_col=0)
    # del raw_file['orgId'], raw_file['classCode'],raw_file['scholarId']
    new_file = pd.merge(reIds, raw_file)
    judgeFile(dirpath + data_config[keyF]['final_path'])
    print('final merge numbers:' + str(len(new_file)))
    new_file.to_csv(dirpath + data_config[keyF]['final_path'])
    
def splitDataUser(raw_path=dirpath + data_config[keyF]['final_path'],
                  train_path=dirpath + data_config[keyF]['train_path'],
                  val_path=dirpath + data_config[keyF]['val_path'],
                  test_path=dirpath + data_config[keyF]['test_path']):
    raw_patents = pd.read_csv(raw_path, index_col=0)
    raw_patents.sort_values(by=['publicationDate'], inplace=True)
    if keyF.endswith('US'):
        raw_patents['date_time'] = pd.to_datetime(raw_patents['publicationDate'], format="%Y%m%d")
    else:
        raw_patents['date_time'] = pd.to_datetime(raw_patents['publicationDate'])
    raw_patents['year'] = raw_patents['date_time'].apply(lambda x: x.year)
    raw_patents['year'] = raw_patents['date_time'].apply(lambda x: x.year)
    
    raw_patents['ts_time'] = raw_patents['date_time'].dt.strftime('%Y%m%d')
    min_ts = int(raw_patents['ts_time'].min())
    raw_patents['ts'] = raw_patents['ts_time'].apply(lambda x: int(x) - min_ts)
    
    del raw_patents['date_time'],raw_patents['ts_time']
    future_datas, validate_datas, historical_datas = [], [], []
    train_years = range(2010, 2018)
    val_years = [2018]
    test_years = [2019]
    for comId, patents in tqdm(raw_patents.groupby(by='orgID')):
        historical_dt = patents[patents['year'].isin(train_years)]
        validat_dt = patents[patents['year'].isin(val_years)]
        future_dt = patents[patents['year'].isin(test_years)]

        historical_datas.append(historical_dt)
        validate_datas.append(validat_dt)
        future_datas.append(future_dt)
    # del future_datas['year'],historical_datas['year']
    future_datas = pd.concat(future_datas, axis=0)
    historical_datas = pd.concat(historical_datas, axis=0)
    validate_datas = pd.concat(validate_datas, axis=0)

    historical_datas.sort_values(by=['publicationDate'], inplace=True)
    validate_datas.sort_values(by=['publicationDate'], inplace=True)
    future_datas.sort_values(by=['publicationDate'], inplace=True)

    del historical_datas['year'], validate_datas['year'], future_datas['year']
    judgeFile(train_path)
    historical_datas.to_csv(train_path, index=False)
    validate_datas.to_csv(val_path, index=False)
    future_datas.to_csv(test_path, index=False)

    print("save train data: "+str(len(historical_datas))+" in " + train_path)
    print("save validate data : "+str(len(validate_datas))+" in " + val_path)
    print("save test data : "+str(len(future_datas))+" in " + test_path)


if __name__ == '__main__':
    ## filterYear()    # 筛选数据2010-2020
    filterPatentYear()
    ## filterCount()   # 每个公司只选400条数据
    filterPatentCount()
    process()
    Reindex()
    reId_all()
    splitDataUser()
    MAP_rawID()     # 相邻层级的映射
    ## adjacentId()
    hierarchy_tree(raw_path=data_config[keyF]["map_hierarchy_tree"], to_path=data_config[keyF]['new_tree'])
    print('TEST')
