from cProfile import label
from dataclasses import dataclass
import matplotlib.pyplot as plt
import os.path

import pandas as pd
from pylab import mpl
dirpath = os.path.dirname(os.path.realpath(__file__)) +'/'
print (dirpath)

# 判断文件是否存在
def judgeFile(path):
    splitP = path.split("/")[:-1]
    dir = "/".join(splitP)
    if not os.path.exists(dir):
        os.mkdir(dir)
        
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']    # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False           # 解决保存图像是负号'-'显示为方块的问题

dicTop = {"A":"人类生活需要","B": "作业、运输","B": "作业、运输","C": "化学；冶金","D":"纺织；造纸","E":"固定建筑", "F":"机械工程、照明、加热、武器、爆破","G":"物理","H":"电学"}
# dfPatent = pd.read_csv('data/copyright/normal_final_patent.csv')
def PatentPivotPlot(df,x,y,xlabel,ylabel,title,grouplist,rot):
    dfGroupedYearLevel = df.groupby(grouplist)[['id']].count()
    dfGroupedDF= pd.DataFrame(dfGroupedYearLevel)
    dfUnStack = dfGroupedDF.unstack()
    # # 获取分组后的内容
    dfGroupedData=dfUnStack['id']
    # 重置索引，将原有year索引置换为列
    dfGroupedData = dfGroupedData.reset_index()
    # 折线图看整个分类对比趋势
    dfGroupedData.plot(x='year',title=title,rot=rot)
    plt.legend(loc='best')
    plt.ylabel(xlabel)
    plt.xlabel(ylabel)
    plt.show()
    # 折线图看单个分类趋势
    titles=[]
    for j in range(0, len(dfGroupedData.columns)-1):
        titles.append('')
    dfGroupedData.plot(x='year',subplots=True, figsize=(6, 8),title=titles,rot=rot)
    plt.legend(loc='best')
    plt.ylabel(xlabel)
    plt.xlabel(ylabel)
    plt.show()
    #柱状图看整个分类数量分布
    dfGroupedData.plot(kind='bar',x=x,title=title, rot=rot)
    plt.legend(loc='best')
    plt.ylabel(xlabel)
    plt.xlabel(ylabel)
    plt.show()
    # 柱状图看单个分类数量分布
    dfGroupedData.plot(kind='bar',x=x,subplots=True,title=titles, figsize=(6, 8), rot=rot)
    plt.legend(loc='best')
    plt.ylabel(xlabel)
    plt.xlabel(ylabel)
    plt.show()

def Patent2DPlot(df,x,y,xlabel,ylabel,title,grouplist,rot):
    dfGrouped = df.groupby(grouplist)[['count']].sum()
    dfGroupedDF= pd.DataFrame(dfGrouped)
    dfGroupedDF = dfGroupedDF.reset_index()
    #柱状图看整个分类数量分布
    dfGroupedDF.plot(kind='bar',x=x,title=title, rot=rot)
    for i in range(0,len(dfGroupedDF.columns)-1):
       plt.text(i,dfGroupedDF.get("count")[i],dfGroupedDF.get("count")[i], ha='center', va='bottom')
    plt.legend(loc='best')
    plt.ylabel(xlabel)
    plt.xlabel(ylabel)
    plt.show()

dfPatent = pd.read_csv(dirpath + 'data/CN-EDA/patents_year_all.csv')

########## By 时间统计专利总数数量
# patentbyTime = dfPatent['patentId'].groupby(dfPatent['year']).count()
# print(patentbyTime.shape)
# print(patentbyTime.head())
# patentbyTimeDF = pd.DataFrame(patentbyTime)
# patentbyTimeDF = patentbyTimeDF.reset_index()
# patentbyTimeDF.rename(columns={"patentId":"count"}, inplace=True)
# judgeFile(dirpath +'data/CN-EDA/patents_year_count.csv')
# patentbyTimeDF.to_csv(dirpath +'data/CN-EDA/patents_year_count.csv')

# patentbyTimeDF.plot(kind='bar', title='patent by year', x='year', y='count', color=['g'])
# for i in range(0,32):
#     plt.text(i,patentbyTimeDF.get("count")[i],patentbyTimeDF.get("count")[i], ha='center', va='bottom')
# plt.ylabel("专利数量")
# plt.xlabel("年份")
# plt.legend(loc='best')
# plt.show()
# exit()


patentbyTime = dfPatent['patentId'].groupby(dfPatent['orgname']).count()
print(patentbyTime.shape)
print(patentbyTime.head())
patentbyTimeDF = pd.DataFrame(patentbyTime)
patentbyTimeDF = patentbyTimeDF.reset_index()
patentbyTimeDF.rename(columns={"patentId":"count"}, inplace=True)

patentbyTimeDF.plot(kind='bar', title='按组织统计专利数量', x='orgname', y='count', color=['blue'],rot=45)
plt.ylabel("专利数量")
plt.xlabel("组织")
plt.legend(loc='best')
plt.show()
exit()
########## By 时间统计专利一级分类数量
### by 时间和分类group by 数量
# dfClassPatent = pd.read_csv('data/copyright/normal_final_class_patent_level1.csv')
# patentClassMerge = pd.merge(dfClassPatent, dfPatent, how='left', on='id')
# print(patentClassMerge.head())
# print(patentClassMerge.shape)
# print(patentClassMerge.columns)
# jointClass = patentClassMerge[['id','Level1','level1name','year','month','gap']]
# jointClass = jointClass.reset_index()
# jointClass.to_csv('data/copyright/final_class_patent_level1.csv')
# patentClassbyTime = dfClassPatent.groupby(['year','level1name'])[['id']].count()
# patentClassbyTimeDF = pd.DataFrame(patentClassbyTime)
# patentClassbyTimeDF.to_csv('data/copyright/report_class_level1name_time.csv')
# print(patentClassbyTimeDF.head())
# print(patentClassbyTimeDF.shape)
# df=patentClassbyTimeDF.unstack()
# print(df.shape)
# print(df.head())
###### 统计专利部总数
# dfTotal = pd.read_csv('data/copyright/normal_final_class_patent_level1.csv')
# patentClassbyTime = dfTotal.groupby(['level1name'])[['id']].count()
# patentClassbyTimeDF = pd.DataFrame(patentClassbyTime)
# patentClassbyTimeDF = patentClassbyTimeDF.reset_index()
# patentClassbyTimeDF=patentClassbyTimeDF.sort_values(by='id', ascending=False)
# patentClassbyTimeDF.reset_index(inplace=True)
# patentClassbyTimeDF2 =patentClassbyTimeDF
# patentClassbyTimeDF2.plot(kind='bar', title='patent by year', x='level1name', y='id', color=['g'],rot=45)
# for i in range(0,8):
#     plt.text(i,patentClassbyTimeDF2.get("id")[i],patentClassbyTimeDF2.get("id")[i], ha='center', va='bottom')
# plt.ylabel("总数")
# plt.xlabel("部")
# plt.legend(loc='best')
# plt.show()
# exit()
# ###  by 时间和一级分类group by 数量绘图
# dfYearClass = pd.read_csv('data/copyright/report_class_level1name_time.csv')
# dfStack = dfYearClass.set_index(['year','level1name'])
# dfUnStack = dfStack.unstack()
# # 获取分组后的内容
# dfGroupedData=dfUnStack['id']
# # 重置索引，将原有year索引置换为列
# dfGroupedData = dfGroupedData.reset_index()
# # 折线图看整个分类对比趋势
# dfGroupedData.plot(x='year',title='The trend of class by year',rot=45)
# plt.legend(loc='best')
# plt.show()
# # 折线图看单个分类趋势
# dfGroupedData.plot(x='year',subplots=True, figsize=(6, 8),title=['','','','','','','',''],rot=45)
# plt.legend(loc='best')
# plt.show()
# #柱状图看整个分类数量分布
# dfGroupedData.plot(kind='bar',x='year',title='The count of class by year', rot=45)
# plt.legend(loc='best') 
# plt.show()
# # 柱状图看单个分类数量分布
# dfGroupedData.plot(kind='bar',x='year',subplots=True,title=['','','','','','','',''], figsize=(6, 8), rot=45)
# plt.legend(loc='best') 
# plt.show()
# exit()

################by 时间统计二级分类
# dfClassPatent = pd.read_csv('data/copyright/normal_final_class_patent_level2.csv')
# patentClassMerge = pd.merge(dfClassPatent, dfPatent, how='left', on='id')
# print(patentClassMerge.head())
# print(patentClassMerge.shape)
# print(patentClassMerge.columns)
# jointClass = patentClassMerge[['id','Level1','level1name','Level2','level2name','year','month','gap']]
# jointClass = jointClass.reset_index()
# jointClass.to_csv('data/copyright/normal_final_class_patent_level2.csv')

# 按照时间统计二级分类保存
# patentClass2byTime = dfYearClass.groupby(['year','level2name'])[['id']].count()
# patentClass2byTimeDF = pd.DataFrame(patentClass2byTime)
# patentClass2byTimeDF.to_csv('data/copyright/report_class_level2name_time.csv')
####  by 具体时间统计专利二级分类，主要统计2017-2020 单独统计 
# dfYearClass = pd.read_csv('data/copyright/normal_final_class_patent_level2.csv')
# for i in  range(2016,2021):
#     for c in list('BFGH'):
#         # by指定年份和指定小类统计
#         dfSubClassByYear=dfYearClass.loc[(dfYearClass['year'] == i) & (dfYearClass['Level1']==c), ["year", "Level2",'id']]
#         countbyClass2 = dfSubClassByYear['id'].groupby(dfSubClassByYear['Level2']).count()
#         patentbyTimeDF = pd.DataFrame(countbyClass2)
#         patentbyTimeDF = patentbyTimeDF.reset_index()
#         patentbyTimeDF.rename(columns={"id":"count"}, inplace=True)
#         patentbyTimeDF.plot(kind='bar', title=str(i)+dicTop[c]+'部下大类数量统计', x='Level2', y='count', color=['blue'])
#         plt.legend(loc='best')
#         plt.show()
        
#         patentbyTimeDF = patentbyTimeDF.sort_values(by='count', ascending=False)
#         top5 = patentbyTimeDF.iloc[0:5]
#         top5 = top5.set_index(['Level2'])
#         top5.plot(kind='pie', subplots=True, figsize=(8, 8),title=str(i)+dicTop[c]+'部下大类Top5分布')
#         print(top5)
#         plt.legend(loc='best')
#         plt.show()
####  by 大类统计2017~2020年增长数量和趋势
# dfSubClass = pd.read_csv('data/copyright/normal_final_class_patent_level2.csv')
# for c in list('BFGH'):
#     dfSubClassByYear = dfSubClass.loc[(dfSubClass['year'] >=2017) & (dfSubClass['year']<=2020) & (dfSubClass['Level1'] == c), ["year", "Level2",'id']]
#     patentClassbyTime = dfSubClassByYear.groupby(['Level2','year'])[['id']].count()
#     patentClassbyTimeDF = pd.DataFrame(patentClassbyTime)
    
#     df=patentClassbyTimeDF.unstack()
#     # 获取分组后的内容
#     dfGroupedData=df['id']
#     # 重置索引，将原有year索引置换为列
#     dfGroupedData = dfGroupedData.reset_index()
#     # 柱状图看整个分类数量分布
#     dfGroupedData.plot(kind='bar',x='Level2',title=dicTop[c]+'部下大类2017~2020数量统计', rot=45)
#     plt.legend(loc='best') 
#     plt.show()
#     # 柱状图看单个分类数量分布
#     dfGroupedData.plot(kind='bar',x='Level2',subplots=True, figsize=(6, 8), rot=45)
#     plt.legend(loc='best') 
#     plt.show()
# exit()

########################公司统计
####### 公司专利发布时间汇总到新文件
# dfCompany = pd.read_csv('data/copyright/normal_final_company_patent.csv')
# dfClass1Patent = pd.read_csv('data/copyright/normal_final_class_patent_level1.csv')
# dfClass2Patent = pd.read_csv('data/copyright/normal_final_class_patent_level2.csv')
# dfCompanyYear = pd.merge(dfCompany, dfPatent, how='left', on='id')
# print(dfCompanyYear.head())
# print(dfCompanyYear.shape)
# print(dfCompanyYear.columns)
# dfCompanyYear = dfCompanyYear[['id','cname','year','month','gap']]
# dfCompanyYear.to_csv('data/copyright/report_compnay_time.csv')

# dfCompanyClass1 = pd.merge(dfCompany, dfClass1Patent, how='left', on='id')
# dfCompanyClass1 = dfCompanyClass1[['id','cname','Level1','level1name','year','month','gap']]
# dfCompanyClass1.to_csv('data/copyright/report_company_class1_time.csv')

# dfCompanyClass2 = pd.merge(dfCompany, dfClass2Patent, how='left', on='id')
# dfCompanyClass2 = dfCompanyClass2[['id','cname','Level1','level1name','Level2','level2name','year','month','gap']]
# dfCompanyClass2.to_csv('data/copyright/report_company_class2_time.csv')
# dfCompany = pd.read_csv('data/copyright/report_compnay_time.csv')
# ######## by时间统计组织发表专利top20
# dfCampanyPatent = dfCompany.groupby(['cname'])[['id']].count()
# dfCampanyPatent = pd.DataFrame(dfCampanyPatent)
# patentbyTimeDF = dfCampanyPatent.sort_values(by='id', ascending=False)
# # top20 = patentbyTimeDF.iloc[0:20]
# # top20=top20.reset_index()
# # top20.plot(kind='bar',subplots=True, title='Company Patent top 20', x='cname', y='id', color=['b'],rot=15)
# # for i in range(0,20):
# #         plt.text(i,top20.get("id")[i],top20.get("id")[i], ha='center', va='bottom')
# # plt.legend(loc='best')
# # plt.show()
# # exit()

# ############## 截取10~20年组织专利Top20增长情况
# dfCompany20 = dfCompany.loc[(dfCompany['year'] >= 2010) & (dfCompany['year'] <= 2020) & (dfCompany['cname'].isin(top20['cname']))]
# dfCampanyPatent20 = dfCompany20.groupby(['cname','year'])[['id']].count()
# dfCampanyPatent20 = pd.DataFrame(dfCampanyPatent20)
# dfUnStack = dfCampanyPatent20.unstack()
# # # 获取分组后的内容
# dfGroupedData=dfUnStack['id']
# # # 重置索引，将原有year索引置换为列
# dfGroupedData = dfGroupedData.reset_index()
# # 柱状图Top20 每年专利增加情况
# dfGroupedData.plot(kind='bar',x='cname',title='', rot=15)
############Top 5 专利分布情况
# top5 = patentbyTimeDF.iloc[0:5]
# top5=top5.reset_index()
# dfCompanyClass1 = pd.read_csv('data/copyright/report_company_class1_time.csv')
# for i in range(0,5):
# # #筛选 top 5
#     dfCompanyFilter =  dfCompanyClass1.loc[(dfCompanyClass1['year'] >= 2010) & (dfCompanyClass1['year'] <= 2020) & (dfCompanyClass1['cname'] == top5['cname'][i])]
#     dfCompanyFilter = dfCompanyFilter[['year','id','level1name']]
#     dfGroupedYearLevel = dfCompanyFilter.groupby(['year','level1name'])[['id']].count()
#     dfGroupedDF= pd.DataFrame(dfGroupedYearLevel)
#     dfUnStack = dfGroupedDF.unstack()
#     # # 获取分组后的内容
#     dfGroupedData=dfUnStack['id']
#     # 重置索引，将原有year索引置换为列
#     dfGroupedData = dfGroupedData.reset_index()
#     # 折线图看整个分类对比趋势
#     dfGroupedData.plot(x='year',title=top5['cname'][i] +'按年统计部分类',rot=45)
#     plt.legend(loc='best')
#     plt.ylabel("专利数量")
#     plt.xlabel("年份")
#     plt.show()
#     # 折线图看单个分类趋势
#     titles=[]
#     for j in range(0, len(dfGroupedData.columns)-1):
#         titles.append('')
#     dfGroupedData.plot(x='year',subplots=True, figsize=(6, 8),title=titles,rot=45)
#     plt.legend(loc='best')
#     plt.ylabel("专利数量")
#     plt.xlabel("年份")
#     plt.show()
#     #柱状图看整个分类数量分布
#     dfGroupedData.plot(kind='bar',x='year',title=top5['cname'][i] +'按年统计部分类', rot=45)
#     plt.legend(loc='best')
#     plt.ylabel("专利数量")
#     plt.xlabel("年份")
#     plt.show()
#     # 柱状图看单个分类数量分布
#     dfGroupedData.plot(kind='bar',x='year',subplots=True,title=titles, figsize=(6, 8), rot=45)
#     plt.legend(loc='best')
#     plt.ylabel("专利数量")
#     plt.xlabel("年份")
#     plt.show()
# exit()
########### Top3 大类分布情况
# top3 = patentbyTimeDF.iloc[0:3]
# top3=top3.reset_index()
# dfCompanyClass1 = pd.read_csv('data/copyright/report_company_class2_time.csv')
# for i in range(0,3):
# # #筛选 top 5
#     dfCompanyFilter =  dfCompanyClass1.loc[(dfCompanyClass1['year'] >= 2015) & (dfCompanyClass1['year'] <= 2020) & (dfCompanyClass1['cname'] == top3['cname'][i])]
#     if i == 0:
#         for c in list('FGH'):
#             dfCompanyFilterLevel = dfCompanyFilter[(dfCompanyFilter['Level1'] == c)]
#             dfCompanyFilterLevel = dfCompanyFilterLevel[['year','id','level2name']]
#             PatentPivotPlot(dfCompanyFilterLevel,'year','','年份','专利大类数量',top3['cname'][i]+'2015~2020'+c+'部大类统计',['year','level2name'],15)
#     if i == 1:
#         for c in list('GHC'):
#             dfCompanyFilterLevel = dfCompanyFilter[(dfCompanyClass1['Level1'] == c)]
#             dfCompanyFilterLevel = dfCompanyFilterLevel[['year','id','level2name']]
#             PatentPivotPlot(dfCompanyFilterLevel,'year','','年份','专利大类数量',top3['cname'][i]+'2015~2020'+c+'部大类统计',['year','level2name'],15)
#     if i == 2:
#         for c in list('HG'):
#             dfCompanyFilterLevel = dfCompanyFilter[(dfCompanyClass1['Level1'] == c)]
#             dfCompanyFilterLevel = dfCompanyFilterLevel[['year','id','level2name']]
#             PatentPivotPlot(dfCompanyFilterLevel,'year','','年份','专利大类数量',top3['cname'][i]+'2015~2020'+c+'部大类统计',['year','level2name'],15)
# exit()



# 添加文字描述
# W = [0.00,0.25,0.50]# 偏移量
# for i in range(20):
#     for a,b in zip(X+W[i],top20.get("id")[i]):#zip拆包
#         plt.text(a,b,"%.0f"% b,ha="center",va= "bottom")#格式化字符串，保留0位小数
# plt.text(0.0,48,"TEXT")# 在（0,48）这个位置，显示TEXT这个值

########################公司统计
####### 公司专利发布时间汇总到新文件
# dfAuthor = pd.read_csv('data/copyright/normal_final_author_patent.csv')
# dfClass1Patent = pd.read_csv('data/copyright/normal_final_class_patent_level1.csv')
# # # dfClass2Patent = pd.read_csv('data/copyright/normal_final_class_patent_level2.csv')
# dfAuthorYear = pd.merge(dfAuthor, dfClass1Patent, how='left', on='id')
# # # print(dfCompanyYear.head())
# # # print(dfCompanyYear.shape)
# # # print(dfCompanyYear.columns)
# # dfAuthorYear = dfAuthorYear[['id','aid','aname','cid','cname','year','month','gap']]
# dfAuthorClass = dfAuthorYear[['id','aid','aname','cid','cname','Level1','level1name','year','month','gap']]

# dfAuthorClass.to_csv('data/copyright/report_author_class_time.csv')

# patentbyAuthor = dfAuthor.groupby(['aid','aname'])[['id']].count()
# patentbyAuthor = pd.DataFrame(patentbyAuthor)
# patentbyAuthor = patentbyAuthor.reset_index()
# patentbyAuthor.rename(columns={"id":"count"}, inplace=True)
# patentbyAuthorSort=patentbyAuthor.sort_values(by='count', ascending=False)
# patentbyAuthor = patentbyAuthorSort.reset_index()
# top5 = patentbyAuthor.iloc[0:5]
# # Patent2DPlot(top5,'aname','','作者','专利数','按作者统计发布专利总数',['aname'],45)

# top5=top5.reset_index()

# dfTop5Author = dfAuthorClass.loc[(dfAuthorClass['aid'].isin(top5['aid']))]
# dfTop5Author = dfTop5Author.groupby(['aname','aid','level1name'])[['id']].count()
# dfTop5Author = pd.DataFrame(dfTop5Author)
# dfUnStack = dfTop5Author.unstack()
# # # 获取分组后的内容
# dfGroupedData=dfUnStack['id']
# # # 重置索引，将原有year索引置换为列
# dfGroupedData = dfGroupedData.reset_index()
# dfGroupedData = dfGroupedData.set_index(['aid'])
# # 柱状图Top20 每年专利增加情况
# dfGroupedData.plot(kind='bar',x='aname',title='', rot=15)
# plt.legend(loc='best')
# plt.ylabel("专利数量")
# plt.xlabel("作者")
# exit()