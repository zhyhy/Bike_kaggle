# ---------------------------------------------- 导入需要的库函数
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor

# ---------------------------------------------- 加载数据集
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
# ---------------------------------------------- 数据清洗
train_WithoutOutliers = train_df[np.abs(train_df['count'] - train_df['count'].mean()) <= 3*train_df['count'].std()]
Bike_data = pd.concat([train_WithoutOutliers,test_df],ignore_index=True)
# 拆分数据
Bike_data['date'] = Bike_data.datetime.apply( lambda c : c.split()[0])
Bike_data['hour'] = Bike_data.datetime.apply( lambda c : c.split()[1].split(':')[0]).astype('int')
Bike_data['year'] = Bike_data.datetime.apply( lambda c : c.split()[0].split('-')[0]).astype('int')
Bike_data['month'] = Bike_data.datetime.apply( lambda c : c.split()[0].split('-')[1]).astype('int')
Bike_data['weekday'] = Bike_data.date.apply( lambda c : datetime.strptime(c, '%Y-%m-%d').isoweekday())
# 观察到windspeed中零居多，1-6则有很多缺失值，可能是补全的，用 RF 对缺失的数据进行预测
Bike_data['windspeed_rfr'] = Bike_data['windspeed']
# 划分数据为 风速为0 风速不为0 两个部分
dataWind0 = Bike_data[Bike_data['windspeed_rfr'] == 0]
dataWindNot0 = Bike_data[Bike_data['windspeed_rfr'] != 0]

# 选择模型
rf_model_wind = RandomForestRegressor(n_estimators=1000, random_state=42)
# 选特征值进行预测缺失的风速
# 考虑风速可能与 季节 天气 湿度 温度 等有关系
windColumns = ['season','weather','humidity','month','year','temp','atemp']
# 将风速不等于 0 的数据作为训练集， fit到RandForestRegressor中
rf_model_wind.fit(dataWindNot0[windColumns],dataWindNot0['windspeed_rfr'])
# 通过训练好的模型预测风速
wind0Values = rf_model_wind.predict(dataWind0[windColumns])
# 将预测好的风速填充到风速为零的数据中
dataWind0.loc[:,'windspeed_rfr'] = wind0Values
# 连接两部分数据
Bike_data = dataWindNot0.append(dataWind0)
Bike_data.reset_index(inplace = True)
Bike_data.drop('index',inplace = True, axis = 1)
# 特征值选择
# 根据前面的观察，将时段（hour) 温度（temp 湿度（humidity) 年份（year） month season weather windspeed_rfr weekday workingday holiday
# 上述11项作为特征值
# 由于CART决策树使用二分类，所以讲多类型数据转换为使用one-hot转化为多个二分型类别
dummies_month = pd.get_dummies(Bike_data['month'],prefix = 'month')
dummies_season = pd.get_dummies(Bike_data['season'],prefix = 'season')
dummies_weather = pd.get_dummies(Bike_data['weather'],prefix='weather')
dummies_year = pd.get_dummies(Bike_data['year'],prefix='year')
# 连接
Bike_data = pd.concat([Bike_data,dummies_month,dummies_season,dummies_weather,dummies_year],axis = 1)

# 分离训练集和测试集
data_train = Bike_data[pd.notnull(Bike_data['count'])]
data_test = Bike_data[~pd.notnull(Bike_data['count'])].sort_values(by=['datetime'])
datetimecol = data_test['datetime']
yLabels = data_train['count']
yLabels_log = np.log(yLabels)


drop_feature = ['casual','count','datetime','date','registered','windspeed','atemp','month','season','weather','year']
data_train = data_train.drop(drop_feature,axis = 1)
data_test = data_test.drop(drop_feature,axis = 1)

# 选择模型，训练模型
rf_model = RandomForestRegressor(n_estimators=1000,random_state=42)
rf_model.fit(data_train,yLabels_log)
preds = rf_model.predict(X = data_train)

# 预测测试集数据
pre_test = rf_model.predict(X = data_test)
submission = pd.DataFrame({'datetime':datetimecol,'count':[max(0,x) for x in np.exp(pre_test)]})
submission.to_csv('predictions.csv',index=False)
