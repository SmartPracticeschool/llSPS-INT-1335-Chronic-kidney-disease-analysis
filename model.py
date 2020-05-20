# -*- coding: utf-8 -*-
"""
Created on Tue May 19 19:14:07 2020

@author: Sanjeev Reddy
"""
import numpy as np
import pandas as pd
from statistics import mode
import sklearn
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from flask import Flask,render_template,url_for,request
import pickle
data=pd.read_csv(r'C:\Users\Sanjeev Reddy\Desktop\chronic_kidney_disease_flask\kidney_disease.csv')
data.drop('id',axis=1,inplace=True)
data['rbc']=data['rbc'].fillna(value=data['rbc'].mode().iloc[0])
data['pc']=data['pc'].fillna(value=data['pc'].mode().iloc[0])
data['pcc']=data['pcc'].fillna(value=data['pcc'].mode().iloc[0])
data['ba']=data['ba'].fillna(value=data['ba'].mode().iloc[0])
data['htn']=data['htn'].fillna(value=data['htn'].mode().iloc[0])
data['dm']=data['dm'].fillna(value=data['dm'].mode().iloc[0])
data['cad']=data['cad'].fillna(value=data['cad'].mode().iloc[0])
data['appet']=data['appet'].fillna(value=data['appet'].mode().iloc[0])
data['pe']=data['pe'].fillna(value=data['pe'].mode().iloc[0])
data['ane']=data['ane'].fillna(value=data['ane'].mode().iloc[0])
data['age']=data['age'].fillna(value=data['age'].mean())
data['bp']=data['bp'].fillna(value=data['bp'].mean())
data['sg']=data['sg'].fillna(value=data['sg'].mean())
data['al']=data['al'].fillna(value=data['al'].mean())
data['su']=data['su'].fillna(value=data['su'].mean())
data['bgr']=data['bgr'].fillna(value=data['bgr'].mean())
data['bu']=data['bu'].fillna(value=data['bu'].mean())
data['sc']=data['sc'].fillna(value=data['sc'].mean())
data['sod']=data['sod'].fillna(value=data['sod'].mean())
data['pot']=data['pot'].fillna(value=data['pot'].mean())
data['hemo']=data['hemo'].fillna(value=data['hemo'].mean())
data['pcv']=data['pcv'].fillna(value=data['pcv'].mode().iloc[0])
data['wc']=data['wc'].fillna(value=data['wc'].mode().iloc[0])
data['rc']=data['rc'].fillna(value=data['rc'].mode().iloc[0])
data.wc=data.wc.replace("\t6200",6200)
data.wc=data.wc.replace("\t8400",8400)
data.wc=data.wc.replace("\t?",8800)
data.pcv=data.pcv.replace("\t?",41)
data.rc=data.rc.replace("\t?",5.2)
data.pcv=data.pcv.astype(int)
data.wc=data.wc.astype(int)
data.rc=data.rc.astype(float)
data.classification=data.classification.replace('ckd\t','ckd')
data.classification=[1 if each=="ckd" else 0 for each in data.classification]
data.rbc=[1 if each=="abnormal" else 0 for each in data.rbc]
data.pc=[1 if each=="abnormal" else 0 for each in data.pc]
data.pcc=[1 if each=="present" else 0 for each in data.pcc]
data.ba=[1 if each=="present" else 0 for each in data.ba]
data.pcc=[1 if each=="present" else 0 for each in data.pcc]
data.htn=[1 if each=="present" else 0 for each in data.htn]
data.dm=[1 if each=="present" else 0 for each in data.dm]
data.cad=[1 if each=="present" else 0 for each in data.cad]
data.appet=[1 if each=="present" else 0 for each in data.appet]
data.pe=[1 if each=="present" else 0 for each in data.pe]
data.ane=[1 if each=="present" else 0 for each in data.ane]
x=data.iloc[:,1:5]
print(x)
y=data.iloc[:,24:]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
dt=DecisionTreeClassifier(criterion='entropy',random_state=0)
dt.fit(x_train,y_train)
pickle.dump(dt,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
print(model.predict([[80,1.02,0,0]]))