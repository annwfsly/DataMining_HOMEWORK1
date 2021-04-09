import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder

# ==============      Okaland Crime     =======================

fpath = "./archive"
f11 = "records-for-2011"
n11 = "new11"
f12 = "records-for-2012"
n12 = "new12"
f13 = "records-for-2013"
n13 = "new13"
f14 = "records-for-2014"
n14 = "new14"
f15 = "records-for-2015"
n15 = "new15"


def FrequentCount(data, fname):
    columns = data.columns.values[1:]

    for clm in columns:
        print("=======  " + fname + " " + clm + " FreqCnt  ========")
        df = data[clm].value_counts()
        BarPlots(df, fname, clm)
        # print(df)


def BarPlots(nums, fname, clm):
    fig = plt.figure(figsize=(15, 7))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    axes = fig.add_subplot(1, 1, 1)
    print(nums)
    nums.iloc[:50].plot(kind='bar', ax=axes, subplots=True, rot=-90)
    axes.set_title(fname + ' ' + clm + ' Histogram')
    axes.set_xlabel(clm)
    # fig.savefig(os.path.join(img_folder, fname + " " + clm + " Histogram.png"))
    plt.show()


def BoxPlots(nums, fname):
    fig, axes = plt.subplots(1, 1)
    color = dict(boxes='DarkGreen', whiskers='DarkOrange',
                 medians='DarkBlue', caps='Red')
    nums.plot(kind='box', ax=axes, subplots=True, color=color, sym='r+')
    axes.set_title(fname + ' Histogram')
    # fig.savefig(os.path.join(img_folder, fname + '_boxplot.png'))
    plt.show()


def fiveNumber(nums):
    # 五数概括 Minimum（最小值）、Q1、Median（中位数、）、Q3、Maximum（最大值）
    Minimum = nums.values.min()  # dataframe格式，需加values
    Maximum = nums.values.max()
    Q1 = np.percentile(nums, 25)
    Median = np.median(nums)
    Q3 = np.percentile(nums, 75)

    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR  # 下限值
    upper_limit = Q3 + 1.5 * IQR  # 上限值

    return Minimum, Q1, Median, Q3, Maximum, lower_limit, upper_limit


def FileOutputs(nums, fname):
    mini, Q1, Medi, Q3, Maxi, l_lim, u_lim = fiveNumber(nums)
    print("===========  " + fname + " Five Numbers   ===========")
    print("minimun:" + str(mini))
    print("quarter1:" + str(Q1))
    print("median:" + str(Medi))
    print("quarter3:" + str(Q3))
    print("lower_limit:" + str(l_lim))
    print("upper_limit:" + str(u_lim))


# =============  缺失值处理  ================
def LostNumProcs(data, meth_flag):
    columns = data.columns
    new_data = data.copy(deep=True)
    totalsam = data.shape[0]
    # print('======  totalsam  =======')
    # print(totalsam)
    if meth_flag == 1:
        # 剔除缺失部分（整行删除）
        print('======  1  =======')
        new_data = data.dropna(inplace=False)
    elif meth_flag == 2:
        # 用最高频率值来填补缺失值
        print('======  2  =======')
        for clm in columns[1:]:
            new_data[clm].fillna(new_data[clm].mode()[0], inplace=True)

    elif meth_flag == 3:
        # 通过属性的相关关系来填补缺失值
        print('======  3  =======')
        miss_clm = []
        comp_clm = []
        for clm in columns[1:]:
            params = data[clm].describe()
            # print('======= params =======')
            # print(int(params[0]))
            if int(totalsam - params[0]) > 0:
                miss_clm.append(clm)
            elif int(totalsam - params[0]) == 0:
                comp_clm.append(clm)
        # print(miss_clm)
        # print(comp_clm)

# miss_index=['country','designation','price','province','region_1','region_2','taster_name','taster_twitter_handle','variety']
# comp_index=['description','points','title','winery']

        # ==========   随机森林 投票决定(reference github: byuegv)   =============
        def set_miss_values(df, complete_index):
            enc_label = OrdinalEncoder()
            enc_fea = OrdinalEncoder()
            missing_index = complete_index[0]

            # Take out the existing numerical data (no NaN) and throw them in Random Forest Regressor
            train_df = df[complete_index]
            # known & unknow values
            known_values = np.array(train_df[train_df[missing_index].notnull()])
            unknow_values = np.array(train_df[train_df[missing_index].isnull()])

            # y is the know missing_index
            y = known_values[:, 0].reshape(-1, 1)
            enc_label.fit(y)
            y = enc_label.transform(y)

            # X are the features
            X = known_values[:, 1:]
            test_X = unknow_values[:, 1:]
            all_X = np.row_stack((X, test_X))
            enc_fea.fit(all_X)
            X = enc_fea.transform(X)

            # fit
            rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
            rfr.fit(X, y.ravel())
            # predict
            predicted_values = rfr.predict(enc_fea.transform(unknow_values[:, 1:]))
            predicted_values = enc_label.inverse_transform(predicted_values.reshape(-1, 1))
            # fill in with predicted values
            df.loc[(df[missing_index].isnull()), missing_index] = predicted_values
            return df

        for i in range(0, len(miss_clm)):
            comp_clm.insert(0, miss_clm[i])
            new_data = set_miss_values(new_data, comp_clm)

    elif meth_flag == 4:
        # 通过数据对象之间的相似性来填补缺失值
        print('======  4  =======')
        numerical_index = ['Area Id', 'Priority']
        # =============  基于KNN算法  ==============
        imputer = KNNImputer(n_neighbors=30)
        imputed_filled = imputer.fit_transform(new_data[numerical_index])
        new_data[numerical_index] = imputed_filled

    return new_data


if __name__ == '__main__':

    # =====  2011,3,5,6 四个数据集归为一类分析  =========
    file_1 = pd.read_csv(os.path.join(fpath, f11 + '.csv'))
    # ===========    打印属性、标签信息   =============
    print("========     info print     =========")
    print("file_2011 info:----------")
    file_1.info()

    # =========     Freq Count     ==============
    FrequentCount(file_1, f11)

    # =========     box plot      =============
    raws_id = file_1[['Area ID']]
    raws_pr = file_1[['Priority']]
    BoxPlots(raws_id, f11)
    BoxPlots(raws_pr, f11)

    # ==============    Five Numbers    ==============
    FileOutputs(raws_id, f11)
    FileOutputs(raws_pr, f11)

    # ================    New Data    ================
    # （分别用四种方法对数据集进行处理、可视化）
    # -----  剔除
    new_f11_1 = LostNumProcs(file_1, 1)
    FrequentCount(new_f11_1, n11 + '-1')

    new_raw1_id = new_f11_1[['Area ID']]
    new_raw1_pr = new_f11_1[['Priority']]
    BoxPlots(new_raw1_id, n11 + '-id-1')
    FileOutputs(new_raw1_pr, n11 + '-id-1')
    BoxPlots(new_raw1_pr, n11 + '-pr-1')
    FileOutputs(new_raw1_pr, n11 + '-pr-1')

    # -----  最高频率填补
    new_f11_2 = LostNumProcs(file_1, 2)

    FrequentCount(new_f11_2, n11 + '-2')

    new_raw2_id = new_f11_2[['Area ID']]
    new_raw2_pr = new_f11_2[['Priority']]
    BoxPlots(new_raw2_id, n11 + '-id-2')
    FileOutputs(new_raw2_id, n11 + '-id-2')
    BoxPlots(new_raw2_pr, n11 + '-pr-2')
    FileOutputs(new_raw2_pr, n11 + '-pr-2')

    # -----  属性相关填补
    new_f11_3 = LostNumProcs(file_1, 3)

    FrequentCount(new_f11_3, n11 + '-3')

    new_raw3_id = new_f11_3[['Area ID']]
    new_raw3_pr = new_f11_3[['Priority']]
    BoxPlots(new_raw3_id, n11 + '-id-3')
    FileOutputs(new_raw3_id, n11 + '-id-3')
    BoxPlots(new_raw3_pr, n11 + '-pr-3')
    FileOutputs(new_raw3_pr, n11 + '-pr-3')

    # -----  相似性填补
    new_f11_4 = LostNumProcs(file_1, 4)
    FrequentCount(new_f11_4, n11 + '-4')

    new_raw4_id = new_f11_4[['points']]
    new_raw4_pr = new_f11_4[['price']]
    BoxPlots(new_raw4_id, n11 + '-id-4')
    FileOutputs(new_raw4_id, n11 + '-id-4')
    BoxPlots(new_raw4_pr, n11 + '-pr-4')
    FileOutputs(new_raw4_pr, n11 + '-pr-4')
