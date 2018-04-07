####DATASET USED CONSIST 1-5000 AS TRAINING DATA AND 5000-5500 AS TESTING DATA
####FOR CONVENIENCE BOTH TRAINING AND TESTING DATA IS COPIED IN ONE FILE ONLY (ss_train.txt)
# IMPORTING LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#READING .txt FILE
dataset = pd.read_csv('ss_train.txt',delimiter ='|')


# TREATING MISSING VALUES 
dataset['13:52:00.1'] = dataset['13:52:00.1'].replace('noTime','00:00:00')


#INDEX NAMING OF DATASET
dataset.rename(columns = {'1':'ID','-0.00197411':'gt_confidence','160':'ground_truth','160.1':'predict_ground_truth','2016-09-11':'date','2016-03-11':'predict_date','13:52:00':'time','13:52:00.1':'predict_time','-1.18041':'tm_confidence','-0.00849628':'dt_confidence'},inplace = True)
dataset.rename(columns = {'-0.00197411':'gt_confidence'},inplace = True)


#CONVERTING DATE AND TIME TO PYTHON DATETIME FORMAT
dataset['date']=pd.to_datetime(dataset.date)
dataset['predict_date']=pd.to_datetime(dataset.predict_date)
day = dataset.date.dt.day
month = dataset.date.dt.month
year = dataset.date.dt.year
pred_day = dataset.predict_date.dt.day
pred_month = dataset.predict_date.dt.month
pred_year = dataset.predict_date.dt.year

dataset['time']=pd.to_datetime(dataset.time)
dataset['predict_time']=pd.to_datetime(dataset.predict_time)
hour = dataset.time.dt.hour
minute = dataset.time.dt.minute
pred_hour = dataset.predict_time.dt.hour
pred_minute = dataset.predict_time.dt.minute

# CREATING DATAFRAME AFTER SPLITTING DATE AND TIME INTO DATE, MONTH, YEAR, HOURS.
df = pd.DataFrame()
df = df.assign(ID=dataset['ID'].values)
df = df.assign(day=day.values)
df = df.assign(month=month.values)
df = df.assign(year=year.values)
df = df.assign(hour=hour.values)
df = df.assign(minute=minute.values)

df = df.assign(pred_day=pred_day.values)
df = df.assign(pred_month=pred_month.values)
df = df.assign(pred_year=pred_year.values)
df = df.assign(pred_hour=pred_hour.values)
df = df.assign(pred_minute=pred_minute.values)

df = df.assign(ground_truth = dataset['ground_truth'].values)
df = df.assign(pred_ground_truth = dataset['predict_ground_truth'].values)
df = df.assign(gt_confidence = dataset['gt_confidence'].values)
df = df.assign(dt_confidence = dataset['dt_confidence'].values)
df = df.assign(tm_confidence = dataset['tm_confidence'].values)



##FOR TASK ONE CALCULATING RESULT WHETHER PREDICTION IS CORRECT(1) OR INCORRECT(0)
df["result"] = ""
for i in range(0,5499) :
    if df['ground_truth'][i] == df['pred_ground_truth'][i] or df['day'][i]==df['pred_day'][i] and df['month'][i]==df['pred_month'][i] and df['year'][i]==df['pred_year'][i] or df['hour'][i]==df['pred_hour'][i] and df['minute'][i]==df['pred_minute'][i]:
      
        df['result'][i] = 1 
    else:
        df['result'][i] = 0
        
df['result'].value_counts()

##FOR BONOUS TASK CALCULATING RESULT WHETHER PREDICTION IS CORRECT(1) OR INCORRECT(0)
df["result"] = ""
for i in range(0,5499) :
    if df['ground_truth'][i] == df['pred_ground_truth'][i] and df['day'][i]==df['pred_day'][i] and df['month'][i]==df['pred_month'][i] and df['year'][i]==df['pred_year'][i] and df['hour'][i]==df['pred_hour'][i] and df['minute'][i]==df['pred_minute'][i]:
      
        df['result'][i] = 1 
    else:
        df['result'][i] = 0
        
df['result'].value_counts()



##6=Pred_day	7=pred_month	8=Pred_year	9=Pred_hour	10=Pred_min	12=Pred_ground_truth	13=gt_confidence	15=dt_confidence	16=tm_confidence
##SPLITTING TRAINING AND TESTING DATA
X = df.iloc[0:4999,[6,7,9,10,12,13,15,16] ].values
y = df.iloc[0:4999, 14:15].values.astype('int')
X_test = df.iloc[4999:5499, [6,7,9,10,12,13,15,16]].values
y_test = df['result'][4999:5499].astype('int')


df['result'].value_counts()
df.describe()

#USING NAIVE BAYES CLASSIFIER MODEL
from sklearn.naive_bayes import GaussianNB    
classifier = GaussianNB()
classifier.fit(X,y)
Y_predict = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,Y_predict)
#efficiency = (322+121)/(53+4+322+121) = .886




#USING RANDOM FOREST CLASSIFIER MODEL
from sklearn.ensemble import RandomForestClassifier
classifier_2 = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier_2.fit(X, y)
Y_predict = classifier_2.predict(X_test).astype('float')
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,Y_predict)
cm
#CHECKING FEATURE IMPORTANCE
classifier_2.feature_importances_

#bonous task
#USING 13
#efficiency1 = (391+77)/(18+14+391+77) = 0.936
#USING 13,15,16
#efficiency2 = (358+102)/(358+102+23+17) = 0.92
#USING 15
#efficiency3 = (355+110)/(335+110+20+15) = 0.96875
#array([ 0.11311374,  0.3746187 ,  0.19034065,  0.04416201,  0.06803098,
       # 0.07144354,  0.13829037])
#USING 16
#efficiency4 = (352+92)/(352+92+23+33) = 0.888
#USING 13,15
#efficiency5 = (356+110)/(356+110+15+19) = 0.932
#USING 13,16
#efficiency6 = (348+87)/(348+87+38+27) = 0.87
#USING 15,16
#efficiency7 = (363+70)/(363+70+55+12) = 0.866



#task 1
#USING ALL FEATURES
#array([[  8,   9],
#       [  2, 481]], dtype=int64)
#profficiency1 = (481+8)/(481+8+9+2) = 0.978
#array([ 0.04044357,  0.03479788,  0.0089407 ,  0.05305092,  0.10420042,
       # 0.07383   ,  0.35860855,  0.20728715,  0.11884082])


#USING ALL FEATURES EXCEPT 3:
#array([[  9,   8],
#       [  0, 483]], dtype=int64)
#array([ 0.04499998,  0.04050885,  0.08073076,  0.09136551,  0.07764796,
#        0.31959038,  0.21363613,  0.13152043])
#PRECISION =  492/500 = 0.984









