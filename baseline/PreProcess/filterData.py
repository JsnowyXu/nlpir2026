import json
import os.path
import re
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch

dirpath = os.path.dirname(os.path.realpath(__file__)) +'/'
print (dirpath)
with open(dirpath+'config/config.json', 'r') as f:
    data_config = json.load(f)

# 判断文件是否存在
def judgeFile(path):
    splitP = path.split("/")[:-1]
    dir = "/".join(splitP)
    if not os.path.exists(dir):
        os.mkdir(dir)

# 过滤掉不属于IPC8个部中的分类号
def classCodeFilter(key):
    def filterC(x):
        classes = x.split(";")
        rule4 = re.compile('^[A-H]\d+[a-zA-Z]-?\d+/\d+')
        catch_clses = []
        for cls in classes:
            catch_cls = rule4.match(cls)
            if catch_cls is None:
                continue
            else:
                catch_clses.append(catch_cls.group())
        if len(catch_clses) == 0:
            return np.NAN
        else:
            return ";".join(catch_clses)
    patents = pd.read_csv(dirpath + data_config[key]['raw_path'], low_memory=False)
    print('original number:' + str(len(patents)))
    patents.dropna(inplace=True)
    print('drop null  number:' + str(len(patents)))
    patents['classCode'] = patents['classCode'].apply(lambda x: filterC(x))
    patents.dropna(inplace=True)
    print('filter class code number:' + str(len(patents)))
    patents['patentId'] = range(len(patents))
    # 去掉所有不符合分类号编码规范的数据
    judgeFile(dirpath + data_config[key]['filter_code_path'])
    
    patents.to_csv(dirpath + data_config[key]['filter_code_path'], index=False)  

def SpliteComPatent(key='US-filter'):
    """
    将专利进行拆分，每行包含一个公司，一个分类号标签
    """
    patents = pd.read_csv(dirpath + data_config[key]['filter_code_path'], low_memory=False)
    patents.dropna(inplace=True)
    com_dict = {}
    for dt_key in data_config[key]['key_COM']:
        com_dict[dt_key] = []
    item = 'orgname' if key.startswith('US') else 'applicantName'
    multicom = 0
    maxcom = 0
    for index, patent in tqdm(patents.iterrows()):
        coms = patent[item].split(";")
        classCodes = patent['classCode'].split(";")
       
        if(len(coms)>1):
            multicom = multicom +1
            if len(coms) > maxcom:
                maxcom = len(coms)
        for compy in coms:
            if len(compy) == 0:
                continue
            for classc in classCodes:
                com_dict['orgname'].append(compy.lower().replace("\"", ""))
                com_dict['classCode'].append(classc)
                com_dict['publicationDate'].append(patent["publicationDate"])
                com_dict['patentId'].append(patent['patentId'])
    com_dict = pd.DataFrame(com_dict)
    print('company 1-1 number:' + str(len(com_dict)))
    print('max multi-com number:' + str(maxcom))
    print('totle multi-com number:' + str(multicom))
    judgeFile(dirpath + data_config[key]['comany_patent_path'])
    com_dict.to_csv(dirpath + data_config[key]['comany_patent_path'])


def filter(topk=30, key='US-filter'):
    # def filterCom(company):
    #     companies = company.split(";")
    #     if len(companies) > 1:
    #         return np.NAN
    #     else:
    #         return companies[0].lower()

    data = pd.read_csv(dirpath + data_config[key]['comany_patent_path'])
    data.dropna(inplace=True)
    # data['patentId'] = range(len(data))
    # item = 'orgname' if key.startswith('US') else 'applicantName'
    # data['orgnameII'] = data[item]
    # data.dropna(inplace=True)
    # del data[item]
    # if key.startswith('CN'):
    #     del data['orgId'], data['scholarId']
    res = []
    for company, patents in tqdm(data.groupby(by='orgname')):
        if len(patents) < topk:
            continue
        else:
            res.append(patents)
    results = pd.concat(res, axis=0)
    # results['orgname'] = results['orgnameII']
    # del results['orgnameII']
    print('filter top 30:' + str(len(results)))

    judgeFile(dirpath + data_config[key]['company_top_path'])
    results.to_csv(dirpath + data_config[key]['company_top_path'], index=False)

# 根据索引id将标签转换为0，1，长度为分类总数
def MultiLabelById(key='US-filter'):
    """
    将分类转化为多标签形式
    """
    def filterC(x):
        classes = x.split(";")
        catch_clses = []
        for cls in classes:
            if cls in catch_clses: continue
            else: catch_clses.append(int(cls))
        label = np.zeros(608)
        label[catch_clses] = 1
        test= ";".join(str(i).split('.')[0] for i in label)
        return test
    patents = pd.read_csv(dirpath + data_config['new-CN']['train_path'], low_memory=False)
    patents.dropna(inplace=True)
    patents['target'] = patents['codeIIIId'].apply(lambda x: filterC(x))
    judgeFile(dirpath + data_config[key]['multi_label_train_path'])
    patents.to_csv(dirpath + data_config[key]['multi_label_train_path'])
    
    patentsval = pd.read_csv(dirpath + data_config['new-CN']['val_path'], low_memory=False)
    patentsval.dropna(inplace=True)
    patentsval['target'] = patentsval['codeIIIId'].apply(lambda x: filterC(x))
    
    judgeFile(dirpath + data_config[key]['multi_label_val_path'])
    patentsval.to_csv(dirpath + data_config[key]['multi_label_val_path'])
    
    patentstest = pd.read_csv(dirpath + data_config['new-CN']['test_path'], low_memory=False)
    patentstest.dropna(inplace=True)
    patentstest['target'] = patentstest['codeIIIId'].apply(lambda x: filterC(x))
    
    judgeFile(dirpath + data_config[key]['multi_label_test_path'])
    patentstest.to_csv(dirpath + data_config[key]['multi_label_test_path'])

def MultiLabel(key='US-filter'):
    """
    将分类转化为多标签形式
    """
    def filterC(x):
        classes = x.split(";")
        rule4 = re.compile('^[A-H]\d+[a-zA-Z]')
        catch_clses = []
        for cls in classes:
            catch_cls = rule4.match(cls)
            if catch_cls is None:
                continue
            else:
                code = catch_cls.group()
                if code in catch_clses: continue
                else: catch_clses.append(code)
        b = pd.Series(catch_clses)
        c = labels.isin(b).astype(int)
        target = c.tolist()
        return ",".join(str(i) for i in target)
    p_labels = pd.read_csv(dirpath + data_config[key]['label_3_path'], low_memory=False)
    labels = pd.Series(p_labels["IPC3"])
    patents = pd.read_csv(dirpath + data_config[key]['COM_path_1'], low_memory=False)
    patents.dropna(inplace=True)
    patents['target'] = patents['classCode'].apply(lambda x: filterC(x))
    judgeFile(dirpath + data_config[key]['multi_label_path'])
    patents.to_csv(dirpath + data_config[key]['multi_label_path'])
    

def buildHierarchy(key='US-filter'):
    data = pd.read_csv(dirpath + data_config[key]['company_top_path'])
    hierar_lbs = data_config[key]['match_label']
    hierar_rules = data_config[key]['match_rule']
    for idx, hierar_key in enumerate(hierar_lbs):
        rule_code = re.compile(hierar_rules[idx])
        data[hierar_key] = data['classCode'].apply(lambda x: rule_code.match(x).group())
        print("finish hierarchical level:{}".format(hierar_key))
    judgeFile(dirpath + data_config[key]['hier_path'])
    ipc = pd.read_csv(dirpath + 'IPC/IPC3.csv')
    data = data[data['codeIII'].isin(ipc['subClass'])]
    data.dropna(inplace=True)
    print('Filter hierarchical number:' + str(len(data)))
    data.to_csv(dirpath + data_config[key]['hier_path'], index=False)


def ReIndex(key='US-filter'):
    patents = pd.read_csv(dirpath + data_config[key]['hier_path'], index_col=0, low_memory=False)
    for rawId, newId, map_path in zip(data_config[key]['reid_raw_col'], data_config[key]['reid_new_col'],
                                      data_config[key]['reid_path']):
        print(rawId, newId, map_path)
        raw_data = patents[rawId].values.tolist()
        if rawId == 'classCode' or rawId =='codeIV':
            raw_data = [string.replace('-', '') for string in raw_data]
        raw_data = list(set(raw_data))
        map_ids = {rawId: raw_data, newId: range(len(raw_data))}
        df_map = pd.DataFrame(map_ids)
        judgeFile(dirpath + map_path)
        print(rawId + ':' + str(len(df_map)))
        df_map.to_csv(dirpath + map_path, index=False)


def IndexFile(key='US-filter'):
    raw_patents = pd.read_csv(dirpath + data_config[key]['hier_path'], index_col=0, low_memory=False)
    reId_paths = data_config[key]['reid_path']
    reId_files = {reid_key: pd.read_csv(dirpath + reid_key) for reid_key in reId_paths}
    new_patents = raw_patents

    print("number of filtering patents {}".format(len(new_patents)))
    for raw_key, (reId_key, reId_file) in zip(data_config[key]['reid_raw_col'], reId_files.items()):
        raw_data = new_patents[raw_key].values.tolist()
        if raw_key == 'classCode' or raw_key =='codeIV':
            new_patents[raw_key] = new_patents[raw_key].apply(lambda x: str(x).replace('-', ''))
        new_patents = pd.merge(left=new_patents, right=reId_file)
        if raw_key !='orgname':
           del new_patents[raw_key]
    judgeFile(dirpath + data_config[key]['reIndex_path'])
    print('reindex number' + str(len(new_patents)))
    new_patents.to_csv(dirpath + data_config[key]['reIndex_path'], index=False)


def process(key='US-filter'):
    # 一个公司对应一个专利和一个分类号，但专利是可重复的，这里需要按照patentid将公司、分类号都合并到一起。
    historical_dt = pd.read_csv(dirpath + data_config[key]['reIndex_path'])
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
        if idx % 100000 == 0:
            dict_pd = pd.DataFrame(dict_dt)
            judgeFile(dirpath + data_config[key]['concate_path'])
            dict_pd.to_csv(dirpath + data_config[key]['concate_path'])
    dict_pd = pd.DataFrame(dict_dt)
    judgeFile(dirpath + data_config[key]['concate_path'])
    dict_pd.to_csv(dirpath + data_config[key]['concate_path'])


def reId_all(key='US-filter'):
    reIds = pd.read_csv(dirpath + data_config[key]['concate_path'], index_col=0)
    # 这里主要是为了去除专利中的文本等消息，找到专利不重复的最初处理完的数据关联即可
    raw_file = pd.read_csv(dirpath + data_config[key]['filter_code_path'], index_col=0)
    del raw_file['orgId'], raw_file['classCode'],raw_file['scholarId']
    new_file = pd.merge(reIds, raw_file)
    judgeFile(dirpath + data_config[key]['final_path'])
    new_file.to_csv(dirpath + data_config[key]['final_path'])


if __name__ == '__main__':
    key = 'US-filter'
    classCodeFilter(key=key)
    # print("---------finish filtering classCode--------------")
    SpliteComPatent(key=key)  # 为保留专利对应的多家公司，按申请公司拆分每行只含专利中的1个公司与1个分类号
    filter(key=key)  # 得到专利数量大于30的公司专利
    ## MultiLabel(key=key)
    print("---------finish filtering--------------")
  
    buildHierarchy(key=key)  # 将分类号逐级列出来
    ReIndex(key=key)  # id重定向
    ### print("---------finish reIndex--------------")
    IndexFile(key=key)  # id映射回源文件
    # # # groupby patentid，将公司，各级分类号都拼接在一起
    ###  process(key=key)  # id映射回源文件
    ### # # 将上一步的csv与原始top30的文件用patentid进行Merge，形成最终的文件
    ### reId_all(key=key)  # 源文件加上文本信息
    ### MultiLabelById(key=key)
    print("---------add text--------------")
