# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 07:09:43 2020

@author: jack
"""
import pandas as pd
import re
import datetime
import matplotlib.pyplot as plt

rdate = re.compile(r'Boletim.*(\d{2}\/\d+\/\d+).* (\d+) casos.* (\d+) recuperados.* (\d+) em isolamento.* (\d+) internado.* (\d+).*bito.*')

rdata = re.compile(r'Boletim.*(\d{2}\/\d+\/\d+).*')
rcasos = re.compile(r'.*\D(\d+) [Cc]asos [Cc]onfirmados.*' )
rrecup = re.compile(r'.*\D(\d+) recuperados.*' )
robito = re.compile(r'.*\D(\d+) [Óó]bito[s]?')
rintern = re.compile(r'.*\D(\d+) internado[s]?')
risol = re.compile(r'.*\D(\d+) em isolamento.*')

#datafile = "table.html"
datafile = "https://www.jaraguadosul.sc.gov.br/boletim-coronavirus"
exportcsv = "cvd-jaragua.csv"
exportpng = "cvd-jaragua.png"

outdf = pd.DataFrame()

df = pd.read_html(datafile, index_col=None, attrs = { 'class': ' table table-bordered table-condensed table-striped'})

#print(df[0][0])

def searchtable(rgx, value):
    try:
        ret = re.search(rgx,value).group(1)
    except:
        ret = None
    return ret
    

outdf['date'] = df[0][0].map(lambda x: datetime.datetime.strptime(str(searchtable(rdata,x)),"%d/%m/%Y") if searchtable(rdata,x) else None)
outdf['casos'] = df[0][0].map(lambda x: int(searchtable(rcasos,x)) if searchtable(rcasos,x) else None)
outdf['isol'] = df[0][0].map(lambda x: int(searchtable(risol,x)) if searchtable(risol,x) else 0)
outdf['recup'] = df[0][0].map(lambda x: int(searchtable(rrecup,x)) if searchtable(rrecup,x) else None)
outdf['dead'] = df[0][0].map(lambda x: int(searchtable(robito,x)) if searchtable(robito,x) else 0)
outdf['intern'] = df[0][0].map(lambda x: int(searchtable(rintern,x)) if searchtable(rintern,x) else 0)

outdf['increm'] = outdf.casos.diff(-1)
outdf['recinc'] = outdf.recup.diff(-1)
outdf = outdf.dropna()
outdf.to_csv(exportcsv,sep=";")
#print(outdf)

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
plt.show()
