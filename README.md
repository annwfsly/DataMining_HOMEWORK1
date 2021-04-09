# DataMining_HOMEWORK1
数据挖掘第一次作业

## 作业要求
1. 问题描述
本次作业中，自行选择2个数据集进行探索性分析与预处理。

2. 数据集
可选数据集包括：

Consumer & Visitor Insights For Neighborhoods
Wine Reviews
Oakland Crime Statistics 2011 to 2016
Chicago Building Violations
Trending YouTube Video Statistics
Melbourne Airbnb Open Data
MLB Pitch Data 2015-2018

3. 数据分析要求

3.1 数据可视化和摘要
数据摘要
      标称属性，给出每个可能取值的频数
      数值属性，给出5数概括及缺失值的个数数据可视化
      使用直方图、盒图等检查数据分布及离群点

3.2 数据缺失的处理
观察数据集中缺失数据，分析其缺失的原因。分别使用下列四种策略对缺失值进行处理:

将缺失部分剔除
用最高频率值来填补缺失值
通过属性的相关关系来填补缺失值
通过数据对象之间的相似性来填补缺失值
注意：在处理后，要可视化地对比新旧数据集。

4. 提交内容
分析过程报告（PDF格式）
程序所在代码仓库地址（建议使用Github或码云），仓库中应包含完整的处理数据的程序和使用说明
所选择的数据集应在仓库的README文件中说明
相关的数据文件不要上传到代码仓库中

## 选择数据集


1. Wine Review

winemag-data_first150k.csv

winemag-data-130k-v2.csv


2.Oakland Crime Statistics 2011 to 2016

records-for-2011.csv

records-for-2012.csv

records-for-2013.csv

records-for-2014.csv

records-for-2015.csv

records-for-2016.csv

## 目录结构
├── Readme.md                   // help

├── OaklandCrime.py             // OaklandCrime数据集分析代码

├── wine_main.py                // wine_main分析代码

├── 数据挖掘第一次作业.ipynb     

└── 数据挖掘第一次作业.pdf
