import mysql.connector
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from collections import Counter
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm, skew
from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNetCV, ElasticNet
from xgboost import XGBRegressor, plot_importance
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import graphviz
from sklearn.preprocessing import StandardScaler
from geopy import distance
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import scale
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from nltk.corpus import stopwords
stop_words = stopwords.words('russian')
import pickle

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("russian")
from nltk.tokenize import word_tokenize
def stemming(text):
      result=''
      text=word_tokenize(text)
      for word in text:
        result=result+' '+stemmer.stem(word)
      return result
20

def sent_to_words_test(sentences):
    clean_texts= []
    for sent in sentences:
        sent = re.sub('\S*@\S*\s?', '', sent)  # remove emails
        sent = re.sub('\s+', ' ', sent)  # remove newline chars
        sent = re.sub("\'", "", sent)  # remove single quotes
        sent = sent.lower() #tolowercase
        sent = re.sub(r'\d+', '', sent) #remove numbers
        sent = sent.strip()
        #sent = gensim.utils.simple_preprocess(str(sent), deacc=True)
        clean_texts.append(sent)
    return clean_texts


knn_reg=pickle.load(open('knn_reg', 'rb'))
xg_reg_numerical=pickle.load(open('xg_reg_numerical', 'rb'))
xg_reg=pickle.load(open('xg_reg', 'rb'))
Meta_ens=pickle.load(open('Meta_ens', 'rb'))
tfidf=pickle.load(open('tfidf', 'rb'))

def form_test_samples():
    test_num_sample = pd.DataFrame(data = [np.zeros(37)] , columns = ['kitchen', 'sq', 'live_sq', 'block', 'kirp', 'monol', 'panel', 'room_1',
       'room_8', 'room_7', 'room_2', 'room_3', 'room_4', 'room_5', 'new_house',
       'old_house', 'd_to_c', 'metro', 'district_Адмиралтейский',
       'district_Василеостровский', 'district_Выборгский',
       'district_Калининский', 'district_Кировский', 'district_Колпинский',
       'district_Красногвардейский', 'district_Красносельский',
       'district_Кронштадтский', 'district_Курортный', 'district_Московский',
       'district_Невский', 'district_Петроградский', 'district_Петродворцовый',
       'district_Приморский', 'district_Пушкинский', 'district_Фрунзенский',
       'district_Центральный', 'room_studio'] )
    test_geo_sample = pd.DataFrame(data = [np.zeros(10)] , columns = ['lon', 'lat', 'block', 'kirp', 'monol', 'old_house', 'room_studio',
       'room_1', 'panel', 'room_2'] )
    print('Введите площадь кухни:')
    kitchen = input()
    test_num_sample['kitchen'][0] = float(kitchen)
    print('Введите площадь все квартиры:')
    sq = input()
    test_num_sample['sq'][0] = float(sq)
    print('Введите площадь жилой площади:')
    live_sq = input()
    test_num_sample['live_sq'][0] = float(live_sq)
    print('Введите тип стен дома(блочный, кирпичный, панельный, моонолитный):')
    walls = input()
    if walls == 'блочный' :
        test_num_sample['block'][0] = 1
        test_geo_sample['block'][0] = 1
    if walls == 'кирпичный' :
        test_num_sample['kirp'][0] = 1
        test_geo_sample['kirp'][0] = 1
    if walls == 'панельный' :
        test_num_sample['panel'][0] = 1
        test_geo_sample['panel'][0] = 1
    if walls == 'моонолитный' :
        test_num_sample['monol'][0] = 1
        test_geo_sample['monol'][0] = 1
    print('Введите количество комнат: (введите 10, если студия)')
    rooms = int(input())
    if rooms == 1:
        test_num_sample['room_1'][0] = 1
        test_geo_sample['room_1'][0] = 1
    if rooms == 2 :
        test_num_sample['room_2'][0] = 1
        test_geo_sample['room_2'][0] = 1
    if rooms == 3 :
        test_num_sample['room_3'][0] = 1
    if rooms == 4 :
        test_num_sample['room_4'][0] = 1
    if rooms == 5 :
        test_num_sample['room_5'][0] = 1
    if rooms == 7 :
        test_num_sample['room_7'][0] = 1
    if rooms == 8 :
        test_num_sample['room_8'][0] = 1
    if rooms == 10 :
        test_num_sample['room_studio'][0] = 1
        test_geo_sample['room_studio'][0] = 1
    print('Введите координаты вашего дома через пробел ( lat lon ):')
    geo = input()
    test_geo_sample['lon'][0]=geo.split()[0]
    test_geo_sample['lat'][0]=geo.split()[1]
    print('Ваш дом новостройка? 1 - если да , 0 - иначе' )
    new_house = int(input())
    if new_house == 0 :
        test_num_sample['old_house'][0] = 1
        test_geo_sample['old_house'][0] = 1
    if new_house == 1 :
        test_num_sample['new_house'][0] = 1
    print('Введите район вашего дома: (например - Адмиралтейский) ')
    district = input()
    if district == 'Адмиралтейский':
        test_num_sample['district_Адмиралтейский'][0] = 1
    if district == 'Василеостровский':
        test_num_sample['district_Василеостровский'][0] = 1
    if district == 'Выборгский':
        test_num_sample['district_Выборгский'][0] = 1
    if district == 'Калининский':
        test_num_sample['district_Калининский'][0] = 1
    if district == 'Кировский':
        test_num_sample['district_Кировский'][0] = 1
    if district == 'Колпинский':
        test_num_sample['district_Колпинский'][0] = 1
    if district == 'Красногвардейский':
        test_num_sample['district_Красногвардейский'][0] = 1
    if district == 'Красносельский':
        test_num_sample['district_Красносельский'][0] = 1
    if district == 'Кронштадтский':
        test_num_sample['district_Кронштадтский'][0] = 1
    if district == 'Курортный':
        test_num_sample['district_Курортный'][0] = 1
    if district == 'Московский':
        test_num_sample['district_Московский'][0] = 1
    if district == 'Петроградский':
        test_num_sample['district_Петроградский'][0] = 1
    if district == 'Невский':
        test_num_sample['district_Невский'][0] = 1
    if district == 'Петродворцовый':
        test_num_sample['district_Петродворцовый'][0] = 1
    if district == 'Приморский':
        test_num_sample['district_Приморский'][0] = 1
    if district == 'Фрунзенский':
        test_num_sample['district_Фрунзенский'][0] = 1
    if district == 'Пушкинский':
        test_num_sample['district_Пушкинский'][0] = 1
    if district == 'Центральный':
        test_num_sample['district_Центральный'][0] = 1
    print('Введите расстояние до метро:' )
    metro = input()
    test_num_sample['metro'][0] = float(metro)
    center = (59.937289, 30.328578)
    point= (geo.split()[0],geo.split()[1])
    d_to_c = distance.distance(center, point).m
    test_num_sample['d_to_c'][0] = d_to_c
    print('Расскажите поподробнее:')
    desc = input()
    features = tfidf.transform(sent_to_words_test([desc]))
    test_text = pd.DataFrame(
        features.todense(),
        columns=tfidf.get_feature_names()
      )
    geo_t, text_t , num_t = test_geo_sample,test_text,test_num_sample
    predicted_from_knn = knn_reg.predict(geo_t)
    predicted_from_text = xg_reg.predict(text_t)
    predicted_from_numbers = xg_reg_numerical.predict(num_t)
    print ('предсказание по геолокации: ' , round(predicted_from_knn[0]*num_t['sq'][0]))
    print ('предсказание по описанию: ' ,round(predicted_from_text[0]*num_t['sq'][0]))
    print ('предсказание по числовым данным: ' ,round(predicted_from_numbers[0]*num_t['sq'][0]))
    geo_cords_t =pd.concat([geo_t['lon'],geo_t['lat']],axis = 1 )
    test_sample = pd.concat([pd.Series(predicted_from_text),
          pd.Series(predicted_from_numbers),
            pd.Series(predicted_from_knn),
                        text_t.reset_index(drop=True),
                        geo_cords_t.reset_index(drop=True),
                        num_t.reset_index(drop=True)],axis =1)
    print ('Полное предсказание: ' ,round(Meta_ens.predict(test_sample)[0]*num_t['sq'][0]))


while True :
    try :
        print('Нажмите Enter, если хотите оценить объект.'
              'Напишите stop, если хотите закончить работу')
        if input() == 'stop':
            break
        form_test_samples()
    except :
        print('Что-то пошло не так, напишите stop. если хотите закончить')
        if input() == 'stop' :
            break

