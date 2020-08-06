# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 07:09:43 2020

@author: jack
"""
import pandas as pd
import re
import datetime
import matplotlib.pyplot as plt

import numpy as np
from datetime import datetime,timedelta
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

def logistic_model(x,a,b,c):
    return c/(1+np.exp(-(x-b)/a))

def exponential_model(x,a,b,c):
    return a*np.exp(b*(x-c))

rdate = re.compile(r'Boletim.*(\d{2}\/\d+\/\d+).* (\d+) casos.* (\d+) recuperados.* (\d+) em isolamento.* (\d+) internado.* (\d+).*bito.*')

rdata = re.compile(r'Boletim.*(\d{2}\/?\d{2}\/?\d{4}).*')
rcasos = re.compile(r'(\d{0,3}\.?\d{1,3})\D*casos confirmados',re.I|re.M )
rrecup = re.compile(r'(\d{0,3}\.?\d{1,3})\D*recuperados' )
robito = re.compile(r'(\d{0,3}\.?\d{1,3}) [Óó]bito[s]?')
rintern = re.compile(r'(\d{0,3}\.?\d{1,3}) internado[s]?')
risol = re.compile(r'(\d{0,3}\.?\d{1,3}) em (isolamento|tratamento)')

#datafile = "table.html"
datafile = "https://www.jaraguadosul.sc.gov.br/boletim-coronavirus"
exportcsv = "cvd-jaragua.csv"
exportpng = "cvd-jaragua.png"
exportpredict = "cvd-jaragua-predict.png"

outdf = pd.DataFrame()

df = pd.read_html(datafile, index_col=None, attrs = { 'class': ' table table-bordered table-condensed table-striped'})

#for j in range(0,15):
#    print(df[0][0][j])
print(df[0][0][8])

def searchtable(rgx, value):
    value = value.replace('.','')
    try:
        ret = re.search(rgx,value).groups()[0]
    except:
#        raise
        ret = 0
    if(ret==None): print(value)
    else: print(ret)
    return ret
    
def getDate(x):
    try:
        return datetime.strptime(str(searchtable(rdata,x)),"%d%m%Y")
    except:
        return datetime.strptime(str(searchtable(rdata,x)),"%d/%m/%Y")

outdf['date'] = df[0][0].map(lambda x: getDate(x) if searchtable(rdata,x) else None)
outdf['casos'] = df[0][0].map(lambda x: int(searchtable(rcasos,x)) if searchtable(rcasos,x) else 0)
outdf['isol'] = df[0][0].map(lambda x: int(searchtable(risol,x)) if searchtable(risol,x) else 0)
outdf['recup'] = df[0][0].map(lambda x: int(searchtable(rrecup,x)) if searchtable(rrecup,x) else None)
outdf['dead'] = df[0][0].map(lambda x: int(searchtable(robito,x)) if searchtable(robito,x) else 0)
outdf['intern'] = df[0][0].map(lambda x: int(searchtable(rintern,x)) if searchtable(rintern,x) else 0)
outdf['dia'] = outdf['date'].map(lambda x: (x - datetime.strptime("06/04/2020","%d/%m/%Y")).days)

outdf['increm'] = outdf.casos.diff(-1)
outdf['recinc'] = outdf.recup.diff(-1)
outdf = outdf.dropna()
#outdf.to_csv(exportcsv,sep=";")
print(outdf)

#Predicao 
x = list(outdf.iloc[:85:-1,6])

y = list(outdf.iloc[:85:-1,1])

## a = infection speed
## b = day of max infection
## c = number of max infected
fit = curve_fit(logistic_model,x,y,p0=[25,370,10000],maxfev=5000)
errors = [np.sqrt(fit[1][i][i]) for i in [0,1,2]]

print("fit",fit[0])

print("error",errors)

a,b,c = fit[0]

sol = int(fsolve(lambda x : logistic_model(x,a,b,c) - int(c),b))

#print("sol",sol)

exp_fit = curve_fit(exponential_model,x,y,p0=[1,1,1],maxfev=5000)



meanspan = 7

fig, (ax1, ax2) = plt.subplots(2, sharex=True)

axs=plt.gca()
axd=plt.gca()

ax1.grid(color='white',linestyle='solid') 
ax1.plot(outdf.date,outdf.casos,'b.-', label='Casos')
ax1.plot(outdf.date,outdf.recup,'g.-', label='Recuperados')
ax1.plot(outdf.date,outdf.isol+outdf.intern,'m.-',label='Ativos')
ax1.plot(outdf.date,outdf.dead,'r.-',label='Óbitos')
ax1.legend()

ax2.plot(outdf.date,outdf.increm,'r*',label='Novos casos')
ax2.plot(outdf.date,outdf.increm.rolling(window=meanspan,min_periods=meanspan).mean().shift(-1*meanspan),'r-',label='Novos casos')
ax2.plot(outdf.date,outdf.recinc,'g*',label='Curados')
ax2.plot(outdf.date,outdf.recinc.rolling(window=meanspan,min_periods=meanspan).mean().shift(-1*meanspan),'g-',label='Curados')

plt.gcf().autofmt_xdate()
plt.legend()
plt.savefig(exportpng,dpi=300)

fig2, ax3 = plt.subplots()
#print(max(x))


pred_x = list(range(int(max(x)),sol))
#ax3.rcParams['figure.figsize'] = [7, 7]
#ax3.rc('font', size=14)# Real data
ax3.scatter(x,y,label="Real data",color="red")# Predicted logistic curve
ax3.plot(x+pred_x, [logistic_model(i,fit[0][0],fit[0][1],fit[0][2]) for i in x+pred_x], label="Logistic model" )# Predicted exponential curve
ax3.plot(x+pred_x, [exponential_model(i,exp_fit[0][0],exp_fit[0][1],exp_fit[0][2]) for i in x+pred_x], label="Exponential model" )
ax3.legend()
ax3.grid(color='black',linestyle='solid') 
#ax3.xlabel("Days since 13 Marco 2020")
#ax3.ylabel("Total number of infected people")
plt.ylim((min(y)*0.9,c*1.1))

plt.axvline(b,color='g')
plt.axvline(sol,color='r')
plt.axhline(c,color='g')
#plt.show()

y_pred_logistic = [logistic_model(i,fit[0][0],fit[0][1],fit[0][2]) for i in x]
y_pred_exp =  [exponential_model(i,exp_fit[0][0], exp_fit[0][1], exp_fit[0][2]) for i in x]
mean_squared_error(y,y_pred_logistic)
mean_squared_error(y,y_pred_exp)





plt.savefig(exportpredict,dpi=300)
plt.show()
