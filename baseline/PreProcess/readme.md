filterData.py文件用于处理原始数据，得到专利数量大于30的公司专利数据。这是原来专利分类的数量，包含的专利分类会减少，可以降低筛选历史专利数量试试。截至到IndexFile方法结束。
classCodeFilter
SpliteComPatent
filter
buildHierarchy

preprocess.py文件得到train,validate和test数据集，以及得到层级树。时间过滤、数量过滤、索引重定向等方法，得到最终数据集信息。这里是处理是保留一篇专利多家申请公司的信息。

filterPatentYear
filterPatentCount
process
Reindex
reId_all
splitDataUser
MAP_rawID

NLP.py文件是通过train.csv得到语料库数据。
config/config.json文件是参数集合。
初始化静态层级树memory


./data_bert/raw-data 存在原始的中英文数据集。
经过处理后的数据存放在"./data/US-filter/new-US/train-val-test/"和"./data/CN-filter/new-CN/train-val-test/"中。