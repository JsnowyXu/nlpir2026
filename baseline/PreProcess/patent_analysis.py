#  random forest
from tkinter import LEFT
import sklearn.ensemble as ensemble
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from collections import Counter
from dateutil.relativedelta import relativedelta
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc, roc_auc_score, precision_score, roc_curve, recall_score, classification_report )

import os.path

dirpath = os.path.dirname(os.path.realpath(__file__)) +'/'
print (dirpath)

# 判断文件是否存在
def judgeFile(path):
    splitP = path.split("/")[:-1]
    dir = "/".join(splitP)
    if not os.path.exists(dir):
        os.mkdir(dir)
        
def MonthGap(earlyDate, lateDate):
    if lateDate > earlyDate:
        gap = relativedelta(lateDate,earlyDate)
        yr = gap.years
        mth = gap.months
        return yr*12 + mth
    else:
        return 0

########### 专利信息处理：增加年份###############
patentDF = pd.read_csv(dirpath + 'data/CN-filter/tmp_hieray.csv')
print(patentDF.shape)
print(patentDF.info())
print(patentDF.dtypes)
# add new feature: publish years, month
patentDF['publishD'] = pd.to_datetime(patentDF['publicationDate'])
patentDF['year'] = patentDF['publishD'].dt.year
print(patentDF.shape)
print(patentDF.columns)
print(patentDF.head())
judgeFile(dirpath +'data/CN-EDA/patents_year_all.csv')
patentDF.to_csv(dirpath +'data/CN-EDA/patents_year_all.csv')


######## 专利分类信息处理 Class: split patent by class(Level 1 and Level 2) #################
# dfPatentClass = pd.read_csv('data/copyright/original_class_pantent.csv')
# print(dfPatentClass.shape)
# splitclass = pd.DataFrame(dfPatentClass.classid.str.split(';').tolist(), index=dfPatentClass.id).stack()
# print(splitclass.head())
# splitclass = splitclass.reset_index()[[0, 'id']] # var1 variable is currently labeled 0
# print(splitclass.head())
# splitclass.columns = ['class', 'id'] # renaming var1
# print(splitclass.shape)
# print(splitclass.head())
# # splitclass.to_csv('data/copyright/class_pantent.csv')

# splitclass['Level1'] = splitclass['class'].apply(lambda x: x[0:1])
# splitclass['Level2'] = splitclass['class'].apply(lambda x: x[0:3])
# splitclass.to_csv('data/copyright/final_class_pantent.csv')

# 专利分类by 一级和二级去掉重复数据
# dfPatentClass = pd.read_csv('data/copyright/final_class_pantent.csv')
# test = dfPatentClass[['id','Level1','Level2']]
# dfdupli1 = test.drop_duplicates(subset=None, keep='first', inplace=False)
# print(dfdupli1.shape)
# dfdupli1.to_csv('data/copyright/final_class_pantent1.csv')

# 专利分类by 一级去掉重复数据
# dfPatentClass = pd.read_csv('data/copyright/final_class_patent_name.csv')
# test = dfPatentClass[['id','Level1','level1name']]
# dfdupli1 = test.drop_duplicates(subset=None, keep='first', inplace=False)
# print(dfdupli1.shape)
# dfdupli1.to_csv('data/copyright/final_class_patent_level1.csv')

# 增加专利分类名称
# dfClass = pd.read_csv('data/copyright/final_class_patent.csv')
# dfClassBase = pd.read_csv('data/copyright/patent_class.csv')
# jointClass=pd.merge(dfClass, dfClassBase, how='left', left_on='Level1', right_on='code')

# print(jointClass.shape)
# print(jointClass.head())
# print(jointClass.info())
# jointClass = jointClass.drop(['code'], axis=1)
# jointClass.rename(columns={"name":"level1name"}, inplace=True)
# print(jointClass.columns)

# jointClass=pd.merge(jointClass, dfClassBase, how='left', left_on='Level2', right_on='code')
# print(jointClass.shape)
# print(jointClass.head())
# print(jointClass.info())
# jointClass = jointClass.drop(['code'], axis=1)
# jointClass.rename(columns={"name":"level2name"}, inplace=True)
# print(jointClass.columns)
# jointClass = jointClass[['id','Level1','level1name','Level2','level2name']]
# jointClass = jointClass.reset_index()
# jointClass.to_csv('data/copyright/final_class_patent_name.csv')

######## 去掉分类专利异常值
# patentDF = pd.read_csv('data/copyright/normal_final_patent.csv')
# dfPatentLevel2 = pd.read_csv('data/copyright/final_class_patent_name.csv')
# print(dfPatentLevel2.shape)
# df_a_filter = dfPatentLevel2[dfPatentLevel2['id'].isin(patentDF['id'])]
# print(df_a_filter.shape)
# df_a_filter.to_csv('data/copyright/normal_final_class_patent_level2.csv')
# dfPatentLevel1 = pd.read_csv('data/copyright/final_class_patent_level1.csv')
# print(dfPatentLevel1.shape)
# df_a_filter = dfPatentLevel1[dfPatentLevel1['id'].isin(patentDF['id'])]
# print(df_a_filter.shape)
# df_a_filter.to_csv('data/copyright/normal_final_class_patent_level1.csv')

#### 重置索引
# patentDF = pd.read_csv('data/copyright/normal_final_author_patent.csv')
# print(patentDF.columns)
# patentDF = patentDF[['id','aid','aname','cid','cname']]
# print(patentDF.shape)
# print(patentDF.head())
# patentDF.reset_index()
# print(patentDF.head())
# patentDF.to_csv('data/copyright/normal_final_author_patent.csv')
#################### 公司专利信息处理
##### 拆分company
# dfPatentCompany = patentDF[['id','company']]
# print(dfPatentCompany.shape)
# splitCompany = pd.DataFrame(dfPatentCompany.company.str.split(',').tolist(), index=dfPatentCompany.id).stack()
# print(splitCompany.head())
# splitCompany = splitCompany.reset_index()[[0, 'id']] # var1 variable is currently labeled 0
# print(splitCompany.head())
# splitCompany.columns = ['company', 'id'] # renaming var1
# print(splitCompany.shape)
# print(splitCompany.head())
# splitCompany['company']=splitCompany['company'].astype('int64')
# splitCompany.to_csv('data/copyright/company_pantent.csv')
##### Merge 公司名称并重命名
# splitCompany=pd.read_csv('data/copyright/company_pantent.csv')
# companyDF = pd.read_csv('data/copyright/zhitu_export_org.csv')
# print(companyDF.shape)
# print(companyDF.info())
# jointCompany = pd.merge(splitCompany, companyDF, how='left', left_on='company', right_on='vid')
# print(jointCompany.shape)
# print(jointCompany.head())
# print(jointCompany.info())
# jointCompany = jointCompany.drop(['company'], axis=1)
# jointCompany.rename(columns={"vid":"cid"}, inplace=True)
# jointCompany.rename(columns={"name":"cname"}, inplace=True)
# print(jointCompany.columns)
# jointCompany.to_csv('data/copyright/final_company_pantent.csv')
######## 去掉公司专利异常值
# patentDF = pd.read_csv('data/copyright/normal_final_patent.csv')
# dfPatentCompany = pd.read_csv('data/copyright/final_company_pantent.csv')
# print(dfPatentCompany.shape)
# df_a_filter = dfPatentCompany[dfPatentCompany['id'].isin(patentDF['id'])]
# print(df_a_filter.shape)
# df_a_filter.to_csv('data/copyright/normal_final_company_patent.csv')
#################### 作者专利信息处理 
# dfPatentAuthor = patentDF[['id','author_list']]
# dfPatentAuthor.to_csv('data/copyright/original_author_pantent.csv')
# dfPatentClass = patentDF[['id','classid']]
# dfPatentClass.to_csv('data/copyright/original_class_pantent.csv')
# divide csv by 100000 and split author to list
# authorDF = pd.read_csv('data/copyright/zhitu_export_author.csv')
# print(authorDF.shape)
# print(authorDF.head())
# for i in range(3, 19):
#     dfPatentAuthor = pd.read_csv('data/copyright/original_author_pantent_'+ str(i) +'.csv')
#     print(dfPatentAuthor.shape)
#     splitAuthor = pd.DataFrame(dfPatentAuthor.author_list.str.split(',').tolist(), index=dfPatentAuthor.id).stack()
#     print(splitAuthor.shape)
#     print(splitAuthor.head())
#     splitAuthor = splitAuthor.reset_index()[[0, 'id']] # var1 variable is currently labeled 0
#     print(splitAuthor.head())
#     splitAuthor.columns = ['author', 'id'] # renaming var1
#     print(splitAuthor.shape)
#     print(splitAuthor.head())
#     splitAuthor['author'] = splitAuthor['author'].astype('int64')
#     splitAuthor.to_csv('data/copyright/author_pantent_'+ str(i) +'.csv')

   
#     jointAuthor = pd.merge(splitAuthor, authorDF, how='left', left_on='author', right_on='vid')
#     print(jointAuthor.shape)
#     print(jointAuthor.columns)
#     print(jointAuthor.head())
#     print(jointAuthor.info())
#     jointAuthor = jointAuthor.drop(['author'], axis=1)
#     jointAuthor.rename(columns={"vid":"aid"}, inplace=True)
#     jointAuthor.rename(columns={"name":"aname"}, inplace=True)
#     jointAuthor.rename(columns={"org_vid":"cid"}, inplace=True)
#     jointAuthor.rename(columns={"org_name":"cname"}, inplace=True)
#     print(jointAuthor.columns)
#     jointAuthor.to_csv('data/copyright/final_author_pantent_'+ str(i) +'.csv')
## 专利作者去除异常值
# patentDF = pd.read_csv('data/copyright/normal_final_patentss.csv')
# dfPatentAuthor = pd.read_csv('data/copyright/final_author_patent.csv')
# print(dfPatentAuthor.shape)
# df_a_filter = dfPatentAuthor[dfPatentAuthor['id'].isin(patentDF['id'])]
# print(df_a_filter.shape)
# df_a_filter.to_csv('data/copyright/normal_final_author_patent.csv')
