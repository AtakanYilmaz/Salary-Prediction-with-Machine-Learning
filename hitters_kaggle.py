
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

# Feature ENGINEERING COLUMNS
# NEW_CAREER_vs_SEASON_HİT = Hits / CHits
# NEW_CAREER_VS_SEASON_ATBAT = AtBat / CAtBat
# NEW_CARRER_VS_SEASON_HMRUN = HmRun / CHmRun
# NEW_CAREER_VS_SEASON_run = Runs / CRuns
# NEW_CAREER_VS_SEASON_rbı = RBI / CRBI
# NEW_CAREER_VS_SEASON_WALKS = Walks / CWAlks
# Years değişkenine göre rookie den proya göre sırala
# NewLeague deşilkenine göre yeni sezonda transfer olup olmadığına değişkne oluştur
# SEASON_SUCCESS_VOLUME(hİTS * hMrUN * rUNS * rbı)
# cARRER_sUCCES_vOLUME (cHİTS * CHMRUN * CRUNS * CRBI)
# No_Hit = atbat - hits
#

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, plot_roc_curve
from sklearn.model_selection import train_test_split, cross_validate

from helpers.data_prep import *
from helpers.eda import *

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df_ = pd.read_csv("Hafta_07/Ders Öncesi Notlar/hitters.csv")
df = df_.copy()

check_df(df) # Salary 59 adet NA değer var. Gerçekten eksik mi yoksa bağlantı kurulabilir mi bak
df.astype({'Salary': 'int64'}).dtypes

cat_cols, num_cols, cat_but_car = grab_col_names(df)

for col in cat_cols:
    cat_summary(df, col)

num_cols = [col for col in num_cols if col not in "Salary"]

for col in num_cols:
    num_summary(df, col) # outliers var gibi duruyor inceleyeceğiz

for col in num_cols:
    print(col, ": ",check_outlier(df, col)) #outliers yok

for col in cat_cols:
    target_summary_with_cat(df, "Salary", col)
    # Division kategorsiinde E'lerin maaşı W'dan daha fazla
    # League A ile League N maaş ortalamaları hemen hemen aynı ( league A %3 daha yüksek)

for col in num_cols:
    target_summary_with_num(df, "Salary", col)

drop_list = high_correlated_cols(df, plot=True)

missing_values_table(df, na_name=False)

df.loc[(df["Salary"].isnull()) & (df["League"]=="A"), "Salary"] = df[~df["Salary"].isnull()].groupby("League").agg({"Salary":"median"})["Salary"]["A"]
df.loc[(df["Salary"].isnull()) & (df["League"]=="N"), "Salary"] = df[~df["Salary"].isnull()].groupby("League").agg({"Salary":"median"})["Salary"]["N"]

#FEature Engineering

df.loc[(df["Years"] < 4), "NEW_YEARS_EXP"] = "rookie"
df.loc[(df["Years"] >= 4) & (df["Years"] < 9), "NEW_YEARS_EXP"] = "pro"
df.loc[(df["Years"] >= 9) & (df["Years"] < 15), "NEW_YEARS_EXP"] = "exprienced"
df.loc[(df["Years"] >= 15), "NEW_YEARS_EXP"] = "old"


df.loc[(df["League"] == "A") & (df["NewLeague"] == "N"), 'NEW_League_TRANS'] = "transfer_A_N"
df.loc[(df["League"] == "N") & (df["NewLeague"] == "A"), 'NEW_League_TRANS'] = "transfer_N_A"
df.loc[(df["League"] == "A") & (df["NewLeague"] == "A"), 'NEW_League_TRANS'] = "stayed_A"
df.loc[(df["League"] == "N") & (df["NewLeague"] == "N"), 'NEW_League_TRANS'] = "stayed_N"



df["NEW_CAREER_vs_SEASON_HİT"] = df["Hits"] / df["CHits"]
df["NEW_CAREER_VS_SEASON_ATBAT"] = df["AtBat"] / df["CAtBat"]
df["NEW_CARRER_VS_SEASON_HMRUN"] = df["HmRun"] / df["CHmRun"]
df["NEW_CAREER_VS_SEASON_RUN"] = df["Runs"] / df["CRuns"]
df["NEW_CAREER_VS_SEASON_RBI"] = df["RBI"] / df["CRBI"]
df["NEW_CAREER_VS_SEASON_WALKS"] = df["Walks"] / df["CWalks"]
df["NEW_SEASON_SUCCESS_VOLUME"] = (df["Hits"] * df["HmRun"] * df["Runs"] * df["RBI"]) / len(df)
df["CAREER_SUCCESS_VOLUME"] = (df["CHits"] * df["CHmRun"] * df["CRuns"] * df["CRBI"]) / len(df)
df["NEW_NO_HIT"] = df["AtBat"] - df["Hits"]

df.fillna(0, inplace=True)
df.isnull().any()
cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if "Salary" not in col]

#############################################
# Label Encoding & Binary Encoding
#############################################

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

for col in binary_cols:
    label_encoder(df, col)

df.head()

#############################################
# 6. One-Hot Encoding
#############################################

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

df = one_hot_encoder(df, ohe_cols, drop_first=True)

df.head()
df.shape

cat_cols, num_cols, cat_but_car = grab_col_names(df)

rare_analyser(df, "Salary", cat_cols)

###################
# RobustScaler: Medyanı çıkar iqr'a böl.
###################

rs = RobustScaler()
df[num_cols] = rs.fit_transform(df[num_cols])

######################################################
# Model Validation: 10-Fold Cross Validation
######################################################


y = df["Salary"]
X = df.drop(["Salary"], axis=1)


log_model = LogisticRegression().fit(X, y)


