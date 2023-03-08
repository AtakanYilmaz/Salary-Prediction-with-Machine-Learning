import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, cross_validate, cross_val_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import  LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
import warnings
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
warnings.simplefilter(action="ignore")

from helpers.data_prep import *
from helpers.eda import *



pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# İŞ PROBLEMİ #

# Maaş bilgileri ve 1986 yılına ait kariyer istatistikleri
# paylaşılan beyzbol oyuncularının maaş tahminleri için
# bir makine öğrenmesi projesi gerçekleştirilebilir mi?

# AtBat: 1986-1987 sezonunda bir beyzbol sopası ile topa yapılan vuruş sayısı
# Hits: 1986-1987 sezonundaki isabet sayısı
# HmRun: 1986-1987 sezonundaki en değerli vuruş sayısı
# Runs: 1986-1987 sezonunda takımına kazandırdığı sayı
# RBI: Bir vurucunun vuruş yaptıgında koşu yaptırdığı oyuncu sayısı
# Walks: Karşı oyuncuya yaptırılan hata sayısı
# Years: Oyuncunun major liginde oynama süresi (sene)
# CAtBat: Oyuncunun kariyeri boyunca topa vurma sayısı
# CHits: Oyuncunun kariyeri boyunca yaptığı isabetli vuruş sayısı
# CHmRun: Oyucunun kariyeri boyunca yaptığı en değerli sayısı
# CRuns: Oyuncunun kariyeri boyunca takımına kazandırdığı sayı
# CRBI: Oyuncunun kariyeri boyunca koşu yaptırdırdığı oyuncu sayısı
# CWalks: Oyuncun kariyeri boyunca karşı oyuncuya yaptırdığı hata sayısı
# League: Oyuncunun sezon sonuna kadar oynadığı ligi gösteren A ve N seviyelerine sahip bir faktör
# Division: 1986 sonunda oyuncunun oynadığı pozisyonu gösteren E ve W seviyelerine sahip bir faktör
# PutOuts: Oyun icinde takım arkadaşınla yardımlaşma
# Assits: 1986-1987 sezonunda oyuncunun yaptığı asist sayısı
# Errors: 1986-1987 sezonundaki oyuncunun hata sayısı
# Salary: Oyuncunun 1986-1987 sezonunda aldığı maaş(bin uzerinden)
# NewLeague: 1987 sezonunun başında oyuncunun ligini gösteren A ve N seviyelerine sahip bir faktör

df_ = pd.read_csv("Hafta_07/Ders Öncesi Notlar/hitters.csv")
df = df_.copy()

check_df(df, head=10)
# 59 adet na değerimiz var. 263 gözlem olduğu için 59 adeti çıkarmak çok sağlıklı olmaz.
# oynadıları liglere göre median ile doldurmak mantıklı geldi bana.

cat_cols, num_cols, cat_but_car = grab_col_names(df)

for col in cat_cols:
    cat_summary(df, col)

for col in num_cols:
    num_summary(df, col)

drop_list = high_correlated_cols(df, plot=True)
#çok yüksek korelasyonluları drop edip ve etmeden deneyeceğiz


low_limit, up_limit = outlier_thresholds(df, num_cols)

for col in num_cols:
    print(col, ": ",check_outlier(df, col))

missing_values_table(df, na_name=True)

df["Salary"].isnull().sum()
###############################################################
# eksik gözlemleri drop edip model başarımıza bakalım. Daha sonra da modelimizde
# eksik verileri doldurup feature engimeerimg yapıp bakalıp

df1 = df.copy()

df1.dropna(inplace=True)
outlier_thresholds(df1, num_cols)

for col in num_cols:
    replace_with_thresholds(df1, col)
check_outlier(df1, num_cols)

df1 = one_hot_encoder(df1, cat_cols)


X = df1.drop("Salary", axis=1)
y = df1["Salary"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=17, test_size=0.20)

reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

reg_model.intercept_ # 109.38076223041162
reg_model.coef_

# Train RMSE
y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred)) # 264
# TRAIN RKARE
reg_model.score(X_train, y_train) #  0.5621695616373135
# Test RMSE
y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred)) # 230.66464405489643
# Test RKARE
reg_model.score(X_test, y_test) #0.6693292131777417
#######################################################################
# ŞİMDİ FEATURE ENGİNERİNG VE OPTİMİZASYON İLE İLERLEYELİM


# salary değşikeninde eksik verileri lige göre median alarak doldurdum
df.loc[(df["Salary"].isnull()) & (df["League"]=="A"), "Salary"] = df[~df["Salary"].isnull()].groupby("League").agg({"Salary":"median"})["Salary"]["A"]
df.loc[(df["Salary"].isnull()) & (df["League"]=="N"), "Salary"] = df[~df["Salary"].isnull()].groupby("League").agg({"Salary":"median"})["Salary"]["N"]

df["Salary"].isnull().sum()
# Eksik veri kalmadığına göre artık feature engineering e başlayabiliriz.

check_df(df)
check_outlier(df, num_cols)

for col in num_cols:
    replace_with_thresholds(df, col)

#bazı gözlemleri "YEAR" değişkeni ile bölersek ortalamalarını bulabiliriz yıllık bazda.Fonksiyonlaştırmayı deneyeleim.

def mean_maker(df, col_name):
    df["NEW_"+col_name+"_MEAN"] = df[col_name] / df["Years"]

mean_maker(df, "CHits")
df.drop

mean_cols = ['CAtBat', 'CHits', 'CHmRun', 'CRuns', 'CRBI']

for col in mean_cols:
    mean_maker(df, col)

sorted(df.Years.unique())

df.loc[(df['Years'] < 4), 'NEW_Years_EXP'] = 'rookie'
df.loc[(df['Years'] >= 4) & (df['Years'] < 9), 'NEW_Years_EXP'] = 'pro'
df.loc[(df['Years'] >= 9) & (df['Years'] < 15), 'NEW_Years_EXP'] = 'experienced'
df.loc[(df['Years'] >= 15), 'NEW_Years_EXP'] = 'old'


df.loc[(df["League"] == "A") & (df["NewLeague"] == "N"), 'NEW_League_TRANS'] = "transfer_A_N"
df.loc[(df["League"] == "N") & (df["NewLeague"] == "A"), 'NEW_League_TRANS'] = "transfer_N_A"
df.loc[(df["League"] == "A") & (df["NewLeague"] == "A"), 'NEW_League_TRANS'] = "stayed_A"
df.loc[(df["League"] == "N") & (df["NewLeague"] == "N"), 'NEW_League_TRANS'] = "stayed_N"

cat_cols, num_cols, cat_but_car = grab_col_names(df)

df = one_hot_encoder(df, cat_cols, drop_first=True)

y = df["Salary"]
X = df.drop(["Salary"], axis=1)

models = [('LR', LinearRegression()),
          ('RF', RandomForestRegressor()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor()),
          ]

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

# RANDOM FORESTS
rf_model = RandomForestRegressor()
rf_random_params = {"max_depth": np.random.randint(5, 50, 10),
                    "max_features": [3, 5, 7, "auto", "sqrt"],
                    "min_samples_split": np.random.randint(2, 50, 20),
                    "n_estimators": [int(x) for x in np.linspace(start=200, stop=1500, num=10)]}

rf_random = RandomizedSearchCV(estimator=rf_model,
                               param_distributions=rf_random_params,
                               n_iter=100,  # denenecek parametre sayısı
                               cv=3,
                               verbose=True,
                               random_state=42,
                               n_jobs=-1)
rf_random.fit(X, y)
rf_random.best_params_




cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

rf_params = {"max_depth": [15,20,30,40,None],
             "max_features": [3, 5, 7,9, "auto"],
             "min_samples_split": [3,5,8,11,13,15],
             "n_estimators": [500,800,1000]}

gbm_params = {"learning_rate": [0.01,0.1,0.001],
              "n_estimators" : [200,500,1000],
              "subsample" : [0.5, 0.7, 1],
              "min_samples_split" : [2, 5, 10],
              "max_depth" : [5,8,10,20]}

xgboost_params = {"learning_rate": [0.1, 0.01, 0.001],
                  "max_depth": [5, 8, 12, 20],
                  "n_estimators": [100, 200, 300, 500],
                  "colsample_bytree": [0.5, 0.8, 1]}

lightgbm_params = {"learning_rate": [0.01, 0.1, 0.001],
                   "n_estimators": [300, 500, 1500],
                   "colsample_bytree": [0.5, 0.7, 1]}


regressors = [("RF", RandomForestRegressor(), rf_params),
              ("GBM", GradientBoostingRegressor(), gbm_params),
              ('XGBoost', XGBRegressor(objective='reg:squarederror'), xgboost_params),
              ('LightGBM', LGBMRegressor(), lightgbm_params)]

best_models = {}



for name, regressor, params in regressors:
    print(f"########## {name} ##########")
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

    gs_best = GridSearchCV(regressor, params, cv=3, n_jobs=-1, verbose=False).fit(X, y)

    final_model = regressor.set_params(**gs_best.best_params_)
    rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE (After): {round(rmse, 4)} ({name}) ")

    print(f"{name} best params: {gs_best.best_params_}", end="\n\n")

    best_models[name] = final_model

voting_reg = VotingRegressor(estimators=[('RF', best_models["RF"]),
                                         ('LightGBM', best_models["LightGBM"])])

voting_reg.fit(X, y)


np.mean(np.sqrt(-cross_val_score(voting_reg, X, y, cv=10, scoring="neg_mean_squared_error")))

