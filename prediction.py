import pandas as pd
import lightgbm as lgb


import datetime
import pandas as pd
class Preparation:
    
    def __init__(self, num, period, lastDate):
        
        self.num = num
        self.period = period
        self.lastDate = lastDate
        self.FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year']
        self.pdf = None
        
    def prepare(self):
        
        self.pdf["date"] = pd.to_datetime(self.pdf['date']) #, format="%d%m%Y")
        self.pdf = self.pdf.set_index(self.pdf["date"])
        
        self.pdf = self.create_features(self.pdf)
        self.pdf = self.pdf.drop(["date"], axis='columns')
        self.pdf = self.pdf[self.FEATURES]
        
            
    def create_features(self, df):
        """
        Create time series features based on time series index.
        """
        df = df.copy()
        df['hour'] = df.index.hour
        df['dayofweek'] = df.index.dayofweek
        df['quarter'] = df.index.quarter
        df['month'] = df.index.month
        df['year'] = df.index.year
        df['dayofyear'] = df.index.dayofyear
        df['dayofmonth'] = df.index.day
        df['weekofyear'] = df.index.isocalendar().week
        
        return df  
    
    def getDaysPeriod(self):
        
        if(self.period=='Monthly'):
            periodChar='M'
            num=int(self.num)*30
        if(self.period=='Daily'):
            periodChar='D'
            num=int(self.num)
        if(self.period=='Weekly'):
            periodChar='W'
            num=int(self.num)*7
        if(self.period=='Yearly'):
            periodChar='Y'
            num=int(self.num)*365
            
        return num, periodChar
    
    
    def create_Data(self):
        
        print("Creating Prediction Data")
        
        lastDate = datetime.date(int(self.lastDate.year), int(self.lastDate.month), int(self.lastDate.day))
        
        days, period = self.getDaysPeriod()
        print(days, type(period), self.lastDate )
        
        
        dates = []

        for i in range(1, int(days)+1):
            nextDate = lastDate + datetime.timedelta(days=i)
            dates.append(nextDate)
            
        self.pdf = pd.DataFrame(dates, columns=['date'])
        self.prepare()
        
        return self.pdf, period
        

class Prediction:
    
    def __init__(self):
        self.FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year']
        self.TARGET = ['total_sales']
        self.model = None
        self.reg_params = None        
    
     
    def create_features(self, df):
        df = df.copy()
        df['hour'] = df.index.hour
        df['dayofweek'] = df.index.dayofweek
        df['quarter'] = df.index.quarter
        df['month'] = df.index.month
        df['year'] = df.index.year
        df['dayofyear'] = df.index.dayofyear
        df['dayofmonth'] = df.index.day
        df['weekofyear'] = df.index.isocalendar().week
        return df  
    
    
    def prepare(self):

        self.path = 'input.csv'
        self.df = pd.read_csv(self.path, parse_dates =["date"])
        
        self.df["date"] = pd.to_datetime(self.df['date']) # format="%d%m%Y")
        self.df = self.df.set_index(self.df["date"])
        self.last_date = sorted(self.df["date"])[-1]
        
        self.df = self.create_features(self.df)
        self.df = self.df.drop(["date"], axis='columns')
        
        
    def train(self):
        
        train = self.df.loc[self.df.index < '2018-05-17']
        test = self.df.loc[self.df.index >= '2018-05-17']
        train = self.create_features(train)
        test = self.create_features(test)
        
        X_train = train[self.FEATURES]
        y_train = train[self.TARGET]

        X_test = test[self.FEATURES]
        y_test = test[self.TARGET]
        
        self.reg_params = {
            'application':'regression_l1',
            'boosting':'gbdt',
                'learning_rate':0.01,
                'num_leaf':20,
                'max_depth':-1,
                'feature_fraction':0.85,
                'reg_alpha':1.2,
                'reg_lambda':3,
                'verbosity':1,
                'bagging_fraction':0.85,
                'bagging_frequency':2,
                }
        self.reg_params['metric'] = ['rmse']

        dtrain = lgb.Dataset(X_train, y_train)
        dval = lgb.Dataset(X_test, y_test)
        self.model = lgb.train(self.reg_params,dtrain, num_boost_round=10000, valid_sets=[dtrain, dval],  verbose_eval=100,early_stopping_rounds=100)#, categorical_feature=['cinema_code'] )
   
    def load_predict(self, num, period):
        
        prd = Preparation(num, period, self.last_date)
        
        pdf, period = prd.create_Data()
        
        pdf["predicted"] = self.model.predict(pdf)
        
        pdf = pdf.resample(period).sum()
        
        out_df = pdf.drop(['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year'], axis='columns')
        out_df.to_csv("output.csv")
        
        print(pdf["predicted"].values)
        
        return pdf["predicted"].values



#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def vis(df):
    df['date'] = pd.to_datetime(df['date'])
    plt.figure()
    plt.plot(df['date'], df['predicted'])
    plt.xlabel('Duration')
    plt.ylabel('Sales')
    plt.title('Sales over time')
    plt.savefig('./webapp/src/assets/result.jpg')


