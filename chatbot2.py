#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
from nltk.chat.util import Chat, reflections
import re
#import random
#import webbrowser
from get_all_tickers import get_tickers as gt
import yfinance as yf
from yahoo_fin import stock_info as si
from flask import Flask, render_template, request
import numpy as np
import wordcloud
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import pandas as pd
import plotly.express as px
from sent_analy import sentiment
from nltk.corpus import subjectivity #From the NLTK corpus we will import subjectivity to classify a tweet as subjective/objective.
from nltk.sentiment import SentimentAnalyzer #SentimentAnalyzer Library to perform the library
from nltk.sentiment.util import *
from nltk.sentiment import vader
from nltk.sentiment.vader import SentimentIntensityAnalyzer #Sentiment and Intensity Analyzer will perform our required analysis
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import os
if not os.path.exists("wordcloud_images"):
    os.mkdir("wordcloud_images")
api_key='0ce8b6189282441e91727a812dc0f110'
from newsapi import NewsApiClient
newsapi = NewsApiClient(api_key=api_key)
from fuzzywuzzy import fuzz
from pandas.io.json import json_normalize
pd.set_option('display.max_colwidth', -1)


# In[2]:



def levenshtein_distance(str1, str2):
     
    '''Aim is to build a 2D matrix and track the cost or changes required
       by comparing each both strings character by character.
    ''' 
    # Initialize the zero matrix  
    row_length = len(str1)+1
    col_length = len(str2)+1
    distance = np.zeros((row_length,col_length),dtype = int)
    
    # Populate matrix of zeros with the indices of each character of both strings
    for i in range(1, row_length):
        for k in range(1,col_length):
            distance[i][0] = i
            distance[0][k] = k
            
    # writng loops to find the cost of deletion, addition and substitution    
    for col in range(1, col_length):
        for row in range(1, row_length):
            if str1[row-1] == str2[col-1]:
                cost = 0 # If the characters are the same in the two strings in a given position [i,j] then the cost is 0
            else:
                cost = 1
                
            distance[row][col] = min(distance[row-1][col] + 1,      # Cost of removal
                                 distance[row][col-1] + 1,          # Cost of addition
                                 distance[row-1][col-1] + cost)     # Cost of substitution
            
    distance = distance[row][col]
    
    return "The strings are {} edits away".format(distance)


# In[3]:


class Muku_Chat(Chat):

    def __init__(self, pairs, reflections={}):

        # add `z` because now items in pairs have three elements
        self._pairs = [(re.compile(x, re.IGNORECASE), y, z) for (x, y, z) in pairs]
        self._reflections = reflections
        self._regex = self._compile_reflections()
        
    def converse(self, quit="quit"):
        user_input = ""
        while user_input != quit:
            user_input = quit
            try:
                user_input = input(">")
            except EOFError as e:
                print(engine.say(e))
            if user_input:
                while user_input[-1] in "!.":
                    user_input = user_input[:-1]
                print(self.respond(user_input))

    def respond(self, str):

        # add `callback` because now items in pairs have three elements
        for (pattern, response, callback) in self._pairs:
            match = pattern.match(str)

            if match:

                resp = random.choice(response)
                resp = self._wildcards(resp, match)

                if resp[-2:] == '?.':
                    resp = resp[:-2] + '.'
                if resp[-2:] == '??':
                    resp = resp[:-2] + '?'

                # run `callback` if exists  
                if callback: # eventually: if callable(callback):
                    callback(match)

                return resp


# In[4]:


import pyttsx3
engine = pyttsx3.init()
engine.getProperty('voices')
engine.setProperty('voice', '1')


# In[5]:


table=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
df=table[0]
newdf=df[["Symbol","Security"]]
dic=newdf.set_index('Symbol')['Security'].to_dict()
df.sample(5)


# In[6]:


def yearly_stock_trend_complete(match):
    s=match.groups()[0]
    d=pd.DataFrame()
    try:
        for (k,v) in dic.items():
            if re.compile(s.lower()).match(v.lower()):
                d=d.append(yf.Ticker(k).history(period='1y',interval='1d').reset_index())
            elif re.compile(s.lower()).match(k.lower()):
                d=d.append(yf.Ticker(k).history(period='1y',interval='1d').reset_index())
        fig = px.line(d, x="Date", y="Close",
                      labels={'Close':'Closing Stock Price'}, 
                      template='plotly_dark',
                     color_discrete_sequence=[ "aqua"],
                      title="Closing Stock Price for the Current Year for "+str(s)
                     )
        fig.write_image("images/"+ str(s) +".png")    
        fig.write_image("images/"+ str(s) +".png")    
        return fig.show()
    except Exception as e: # work on python 3.x
        engine.say(str(e))       


# In[7]:


def five_yearly_stock_trend_complete(match):
    s=match.groups()[0]
    d=pd.DataFrame()
    try:
        for (k,v) in dic.items():
            if re.compile(s.lower()).match(v.lower()):
                d=d.append(yf.Ticker(k).history(period='5y',interval='1wk').reset_index())
            elif re.compile(s.lower()).match(k.lower()):
                d=d.append(yf.Ticker(k).history(period='5y',interval='1wk').reset_index())
        fig = px.line(d, x="Date", y="Close",
                      labels={'Close':'Closing Stock Price'}, 
                      template='plotly_dark',
                     color_discrete_sequence=[ "aqua"],
                      title="Closing Stock Price for the Last 5 years for "+str(s)
                     )
        fig.write_image("images/"+ str(s) +".png")    
        fig.write_image("images/"+ str(s) +".png")    
        return fig.show()
    except Exception as e: # work on python 3.x
        engine.say(str(e))
        


# In[8]:


def monthly_stock_trend_complete(match):
    s=match.groups()[0]
    d=pd.DataFrame()
    try:
        for (k,v) in dic.items():
            if re.compile(s.lower()).match(v.lower()):
                d=d.append(yf.Ticker(k).history(period='1mo',interval='1d').reset_index())
            elif re.compile(s.lower()).match(k.lower()):
                d=d.append(yf.Ticker(k).history(period='1mo',interval='1d').reset_index())
        fig = px.line(d, x="Date", y="Close",
                      labels={'Close':'Closing Stock Price'}, 
                      template='plotly_dark',
                     color_discrete_sequence=[ "aqua"],
                      title="Closing Stock Price for the Current Year for "+str(s)
                     )
        print(fig.show())
        fig.write_image("images/"+ str(s) +".png")    
    except Exception as e: # work on python 3.x
        engine.say(str(e))


# In[9]:


def sentiment_d(match):
    s=match.groups()[0]
    wordcloud = WordCloud().generate(s)
    fig, axes= plt.subplots(figsize=(20,12),clear=True)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.show()
    wordcloud.to_file("wordcloud_images/"+ str(s[0:7]) +".png")


# In[10]:


def dividends(match):
    s=match.groups()[0]
    try:
        for (k,v) in dic.items():
            if re.compile(s.lower()).match(v.lower()):
                m=yf.Ticker(k)
                st="Dividends info of "+str(v).upper()+" is:\n"+ str(m.dividends)
                print(st)
                engine.say(st)
                engine.runAndWait()
                engine.stop()
            elif re.compile(s.lower()).match(k.lower()):
                m=yf.Ticker(k)
                st="Dividends info of "+str(v).upper()+" is:\n"+ str(m.dividends)
                print(st)
                engine.say(st)
                engine.runAndWait()
                engine.stop()
    except Exception as e: # work on python 3.x
        engine.say(str(e))
        


# In[11]:


def actions(match):
    s=match.groups()[0]
    try:
        for (k,v) in dic.items():
            if re.compile(s.lower()).match(v.lower()):
                m=yf.Ticker(k)
                print("actions info of "+str(v).upper()+" is:\n"+ str(m.actions))
            elif re.compile(s.lower()).match(k.lower()):
                m=yf.Ticker(k)
                print("actions info of "+str(v).upper()+" is: \n"+ str(m.actions))
    except Exception as e: # work on python 3.x
        engine.say(str(e))


# In[12]:


def options(match):
    s=match.groups()[0]
    try:
        for (k,v) in dic.items():
            if re.compile(s.lower()).match(v.lower()):
                m=yf.Ticker(k)
                print("options info of "+str(v).upper()+" is:\n"+ str(m.options))
            elif re.compile(s.lower()).match(k.lower()):
                m=yf.Ticker(k)
                print("options info of "+str(v).upper()+" is: \n"+ str(m.options))
    except Exception as e: # work on python 3.x
        engine.say(str(e))


# In[13]:


def isin(match):
    s=match.groups()[0]
    try:
        for (k,v) in dic.items():
            if re.compile(s.lower()).match(v.lower()):
                m=yf.Ticker(k)
                print("International Securities Identification Number of "+str(v).upper()+" is:\n"+ str(m.isin))
            elif re.compile(s.lower()).match(k.lower()):
                m=yf.Ticker(k)
                print("International Securities Identification Number of "+str(v).upper()+" is: \n"+ str(m.isin))
    except Exception as e: # work on python 3.x
        engine.say(str(e))
        


# In[14]:


def financials(match):
    s=match.groups()[0]
    try:
        for (k,v) in dic.items():
            if re.compile(s.lower()).match(v.lower()):
                m=yf.Ticker(k)
                print("Financials info of "+str(v).upper()+" is: \n"+ str(m.financials))
            elif re.compile(s.lower()).match(k.lower()):
                m=yf.Ticker(k)
                print("Financials of "+str(v).upper()+" is: \n"+ str(m.financials))
    except Exception as e: # work on python 3.x
        engine.say(str(e))


# In[15]:


def major_holders(match):
    s=match.groups()[0]
    try:
        for (k,v) in dic.items():
            if re.compile(s.lower()).match(v.lower()):
                m=yf.Ticker(k)
                print("Major Holders info of "+str(v).upper()+" is: \n"+ str(m.major_holders))
            elif re.compile(s.lower()).match(k.lower()):
                m=yf.Ticker(k)
                df=pd.DataFrame(m.major_holders,columns=['Percentage','Held_By'])
                print("Major Holders info of "+str(v).upper()+" is: \n"+ str(m.major_holders))
    except Exception as e: # work on python 3.x
        engine.say(str(e))


# In[16]:


def quarterly_financials(match):
    s=match.groups()[0]
    try:
        for (k,v) in dic.items():
            if re.compile(s.lower()).match(v.lower()):
                m=yf.Ticker(k)
                print("Quarterly Financials info of "+str(v).upper()+" is: \n"+ str(m.quarterly_financials))
            elif re.compile(s.lower()).match(k.lower()):
                m=yf.Ticker(k)
                print("Quarterly Financials of "+str(v).upper()+" is: \n"+ str(m.quarterly_financials))
    except Exception as e: # work on python 3.x
        engine.say(str(e))


# In[17]:


def balance_sheet(match):
    s=match.groups()[0]
    try:
        for (k,v) in dic.items():
            if re.compile(s.lower()).match(v.lower()):
                m=yf.Ticker(k)
                print("Balance Sheet info of "+str(v).upper()+" is: \n"+ str(m.balance_sheet))
            elif re.compile(s.lower()).match(k.lower()):
                m=yf.Ticker(k)
                print("Balance Sheet info of "+str(v).upper()+" is: \n"+ str(m.balance_sheet))
    except Exception as e: # work on python 3.x
        engine.say(str(e))


# In[18]:


def cashflow(match):
    s=match.groups()[0]
    try:
        for (k,v) in dic.items():
            if re.compile(s.lower()).match(v.lower()):
                m=yf.Ticker(k)
                print("cashflow info of "+str(v).upper()+" is: \n"+ str(m.cashflow))
            elif re.compile(s.lower()).match(k.lower()):
                m=yf.Ticker(k)
                print("cashflow info of "+str(v).upper()+" is: \n"+ str(m.cashflow))
    except Exception as e: # work on python 3.x
        engine.say(str(e))


# In[19]:


def earnings(match):
    s=match.groups()[0]
    try:
        for (k,v) in dic.items():
            if re.compile(s.lower()).match(v.lower()):
                m=yf.Ticker(k)
                print("earnings info of "+str(v).upper()+" is: \n"+ str(m.earnings))
            elif re.compile(s.lower()).match(k.lower()):
                m=yf.Ticker(k)
                print("earnings info of "+str(v).upper()+" is: \n"+ str(m.earnings))
    except Exception as e: # work on python 3.x
        engine.say(str(e))


# In[20]:


def sustainability(match):
    s=match.groups()[0]
    try:
        for (k,v) in dic.items():
            if re.compile(s.lower()).match(v.lower()):
                m=yf.Ticker(k)
                print("sustainability info of "+str(v).upper()+" is: \n"+ str(m.sustainability))
            elif re.compile(s.lower()).match(k.lower()):
                m=yf.Ticker(k)
                print("sustainability info of "+str(v).upper()+" is: \n"+ str(m.sustainability))
    except Exception as e: # work on python 3.x
        engine.say(str(e))


# In[21]:


def recommendations(match):
    s=match.groups()[0]
    try:
        for (k,v) in dic.items():
            if re.compile(s.lower()).match(v.lower()):
                m=yf.Ticker(k)
                recom=pd.DataFrame(m.recommendations).reset_index()
                recom.to_csv("recommendations_of_"+str(v).upper()+".csv",index=False)
                print("recommendations info of "+str(v).upper()+" is: \n"+ str(m.recommendations))
            elif re.compile(s.lower()).match(k.lower()):
                m=yf.Ticker(k)
                recom=pd.DataFrame(m.recommendations).reset_index()
                recom.to_csv("recommendations_of_"+str(v).upper()+".csv",index=False)
                print("recommendations info of "+str(v).upper()+" is: \n"+ str(m.recommendations))
    except Exception as e: # work on python 3.x
        engine.say(str(e))


# In[22]:


def calendar(match):
    s=match.groups()[0]
    try:
        for (k,v) in dic.items():
            if re.compile(s.lower()).match(v.lower()):
                m=yf.Ticker(k)
                print("calendar info of "+str(v).upper()+" is: \n"+ str(m.calendar))
            elif re.compile(s.lower()).match(k.lower()):
                m=yf.Ticker(k)
                print("calendar info of "+str(v).upper()+" is: \n"+ str(m.calendar))
    except Exception as e: # work on python 3.x
        engine.say(str(e))
        


# In[23]:


def quarterly_cashflow(match):
    s=match.groups()[0]
    try:
        for (k,v) in dic.items():
            if re.compile(s.lower()).match(v.lower()):
                m=yf.Ticker(k)
                print("Quarterly cashflow info of "+str(v).upper()+" is: \n"+ str(m.quarterly_cashflow))
            elif re.compile(s.lower()).match(k.lower()):
                m=yf.Ticker(k)
                print("Quarterly cashflow info of "+str(v).upper()+" is: \n"+ str(m.quarterly_cashflow))
    except Exception as e: # work on python 3.x
        engine.say(str(e))
        


# In[24]:


def quarterly_earnings(match):
    s=match.groups()[0]
    try:
        for (k,v) in dic.items():
            if re.compile(s.lower()).match(v.lower()):
                m=yf.Ticker(k)
                print("Quarterly earnings info of "+str(v).upper()+" is: \n"+ str(m.quarterly_earnings))
            elif re.compile(s.lower()).match(k.lower()):
                m=yf.Ticker(k)
                print("Quarterly earnings info of "+str(v).upper()+" is: \n"+ str(m.quarterly_earnings))
    except Exception as e: # work on python 3.x
        engine.say(str(e))


# In[25]:


def quarterly_balance_sheet(match):
    s=match.groups()[0]
    try:
        for (k,v) in dic.items():
            if re.compile(s.lower()).match(v.lower()):
                m=yf.Ticker(k)
                print("Quarterly Balance Sheet info of "+str(v).upper()+" is: \n"+ str(m.quarterly_balance_sheet))
            elif re.compile(s.lower()).match(k.lower()):
                m=yf.Ticker(k)
                print("Quarterly Balance Sheet info of "+str(v).upper()+" is: \n"+ str(m.quarterly_balance_sheet))
    except Exception as e: # work on python 3.x
        engine.say(str(e))


# In[26]:


def institutional_holders(match):
    s=match.groups()[0]
    try:   
        for (k,v) in dic.items():
            if re.compile(s.lower()).match(v.lower()):
                m=yf.Ticker(k)
                print("Institutional Holders info of "+str(v).upper()+" is: \n"+ str(m.institutional_holders))
            elif re.compile(s.lower()).match(k.lower()):
                m=yf.Ticker(k)
                print("Institutional Holders info of "+str(v).upper()+" is: \n"+ str(m.institutional_holders))
    except Exception as e: # work on python 3.x
        engine.say(str(e))
        


# In[27]:


def splits(match):
    s=match.groups()[0]
    try:
        for (k,v) in dic.items():
            if re.compile(s.lower()).match(v.lower()):
                m=yf.Ticker(k)
                print("Splits info of "+str(v).upper()+" is: \n"+ str(m.splits))
            elif re.compile(s.lower()).match(k.lower()):
                m=yf.Ticker(k)
                print("Splits info of "+str(v).upper()+" is: \n"+ str(m.splits))
    except Exception as e: # work on python 3.x
        engine.say(str(e))


# In[28]:


from yahoo_fin.stock_info import *


# In[29]:


from yahoo_fin.stock_info import *
def analyst_info(match):
    s=match
    for (k,v) in dic.items():
        #
        if re.compile(s.lower()).match(v.lower()):
            df=pd.DataFrame(get_analysts_info(k))
            print(df.head(2))
            #print("Analyst info of "+str(v).upper()+" is: \n"+ str(get_analysts_info(k)))
        elif re.compile(s.lower()).match(k.lower()):
            print("Analyst info of "+str(v).upper()+" is: \n"+ str(get_analysts_info(k)))
            #df=pd.DataFrame.from_dict(get_analysts_info(k),columns=['Earnings Estimate',''])
            #print(df.head(2))
            
#analyst_info('AAPL')


# In[30]:


from yahoo_fin.stock_info import *
def analyst_info(match):
    s=match.groups()[0]
    print(s)
    for (k,v) in dic.items():
        if re.compile(s.lower()).match(v.lower()):
            print("Analyst info of "+str(v).upper()+" is: \n"+ str(get_analysts_info(k)))
        elif re.compile(s.lower()).match(k.lower()):
            print("Analyst info of "+str(v).upper()+" is: \n"+ str(get_analysts_info(k)))


# In[31]:


def day_losers(match):
    s=match.groups()[0]
    try:
        if s:
            df=pd.DataFrame(get_day_losers())
            df.to_csv('stockbot/day_losers.csv',index=False)
            #wordcloud.to_file("wordcloud_images/"+ str(s[0:7]) +".png")

            print(df.head())
    except Exception as e: # work on python 3.x
        engine.say(str(e))
        


# In[32]:


def day_gainers(match):
    s=match.groups()[0]
    try:
        if s:
            df=pd.DataFrame(get_day_gainers())
            df.to_csv('stockbot/day_gainers.csv',index=False)
            #wordcloud.to_file("wordcloud_images/"+ str(s[0:7]) +".png")

            print(df.head())
    except Exception as e: # work on python 3.x
        engine.say(str(e))
        


# In[33]:


def day_most_active(match):
    s=match.groups()[0]
    try:
        if s:
            df=pd.DataFrame(get_day_most_active())
            df.to_csv('stockbot/day_most_active.csv',index=False)
            #wordcloud.to_file("wordcloud_images/"+ str(s[0:7]) +".png")

            st=df.head()
            print(st)
            engine.say(st)
            engine.runAndWait()
            engine.stop()
    except Exception as e: # work on python 3.x
        engine.say(str(e))
        


# In[34]:


def day_stats(match):
    s=match.groups()[0]
    try:
        for (k,v) in dic.items():
            if re.compile(s.lower()).match(v.lower()):
                print("Stats info of "+str(v).upper()+" is: \n"+ str(get_stats(k)))
            elif re.compile(s.lower()).match(k.lower()):
                print("Stats info of "+str(v).upper()+" is: \n"+ str(get_stats(k)))
            df=pd.DataFrame(get_stats())
            df.to_csv('stockbot/day_stats.csv',index=False)
            #wordcloud.to_file("wordcloud_images/"+ str(s[0:7]) +".png")

            st=df.head()
            print(st)
            engine.say(st)
            engine.runAndWait()
            engine.stop()
    except Exception as e: # work on python 3.x
        engine.say(str(e))
        


# In[35]:


def day_stats_valuation(match):
    s=match.groups()[0]
    try:
        for (k,v) in dic.items():
            if re.compile(s.lower()).match(v.lower()):
                print("Stats info of "+str(v).upper()+" is: \n"+ str(get_stats_valuation(k)))
            elif re.compile(s.lower()).match(k.lower()):
                print("Stats info of "+str(v).upper()+" is: \n"+ str(get_stats_valuation(k)))
            df=pd.DataFrame(get_stats())
            df.to_csv('stockbot/day_stats.csv',index=False)
            #wordcloud.to_file("wordcloud_images/"+ str(s[0:7]) +".png")

            st=df.head()
            print(st)
            engine.say(st)
            engine.runAndWait()
            engine.stop()    
    except Exception as e: # work on python 3.x
        engine.say(str(e))
        


# In[36]:


def yearly_stock_compare(match):
    try:
        s=match.groups()[0]
        s1=match.groups()[1]
        s2=match.groups()[2]
        s3=match.groups()[3]
        s4=match.groups()[4]
        d=pd.DataFrame()
        d1=pd.DataFrame()    
        d2=pd.DataFrame()    
        d3=pd.DataFrame()
        d4=pd.DataFrame()
        df12=pd.DataFrame()
        d123=pd.DataFrame()    
        d1234=pd.DataFrame()    
        d12345=pd.DataFrame()


        df_vis=pd.DataFrame()
        import matplotlib.animation as ani
        for (k,v) in dic.items():
            if re.compile(s.lower()).match(v.lower()):
                d=d.append(yf.Ticker(k).history(period='1y',interval='1d').reset_index())
            elif re.compile(s.lower()).match(k.lower()):
                d=d.append(yf.Ticker(k).history(period='1y',interval='1d').reset_index())
                d['companyName']=v
                #print(d.info())
        for (k,v) in dic.items():
            if re.compile(s1.lower()).match(v.lower()):
                d1=d1.append(yf.Ticker(k).history(period='1y',interval='1d').reset_index())
            elif re.compile(s1.lower()).match(k.lower()):
                d1=d1.append(yf.Ticker(k).history(period='1y',interval='1d').reset_index())
                d1['companyName']=v

        for (k,v) in dic.items():
            if re.compile(s2.lower()).match(v.lower()):
                d2=d2.append(yf.Ticker(k).history(period='1y',interval='1d').reset_index())
            elif re.compile(s2.lower()).match(k.lower()):
                d2=d2.append(yf.Ticker(k).history(period='1y',interval='1d').reset_index())
                d2['companyName']=v
        for (k,v) in dic.items():
            if re.compile(s3.lower()).match(v.lower()):
                d3=d3.append(yf.Ticker(k).history(period='1y',interval='1d').reset_index())
            elif re.compile(s3.lower()).match(k.lower()):
                d3=d3.append(yf.Ticker(k).history(period='1y',interval='1d').reset_index())
                d3['companyName']=v            

        for (k,v) in dic.items():
            if re.compile(s4.lower()).match(v.lower()):
                d4=d4.append(yf.Ticker(k).history(period='1y',interval='1d').reset_index())
            elif re.compile(s4.lower()).match(k.lower()):
                d4=d4.append(yf.Ticker(k).history(period='1y',interval='1d').reset_index())
                d4['companyName']=v

                newlist=[d,d1,d2,d3,d4]
                df1234=pd.concat(newlist,ignore_index=True)
                df_vis=df_vis.append(df1234)
        df_vis['companyName']=df_vis['companyName'].fillna('Something else')
        fig4 = px.line(df_vis, x="Date", y="Close",color="companyName",
                      labels={'Close':'Closing Stock Price'}, 
                      template='plotly_dark',
                      title="Closing Stock Price for the Current Year for "
                     )
        fig4.write_image("images/new" +".png")    
        print(fig4.show())
    except Exception as e: # work on python 3.x
        engine.say(str(e))
        


# In[37]:


import pyttsx3
engine = pyttsx3.init()
engine.getProperty('voice')
engine.setProperty('voice', '1')
rate = engine.getProperty('rate')
engine.setProperty('rate', rate+20)


# In[39]:


def stock(match):
    s=match.groups()[0]
    d=[]
    try:
        for (k,v) in dic.items():
            if re.compile(s.lower()).match(v.lower()):
                st="Current stock price of "+str(v).upper()+" is: "+ str(np.round(si.get_live_price(k),2))
                print(st)
                engine.say(st)
                engine.runAndWait()
                engine.stop()
            elif re.compile(s.lower()).match(k.lower()):
                st="Current stock price of "+str(v).upper()+" is: "+ str(np.round(si.get_live_price(k),2))
                print(st)
                engine.say(st)
                engine.runAndWait()
                engine.stop()
           
    except Exception as e: # work on python 3.x
        engine.say(str(e))
    


# In[40]:


def sentiment(match):
    txt=match.groups()[0]
    senti = SentimentIntensityAnalyzer()
    try:
        if (fuzz.token_set_ratio(txt,"document")>50) or (fuzz(txt,"file")>50):
            file_path=input("enter file path: ")
            f= open(file_path)
            f1=str(f.readlines())
            sia=senti.polarity_scores(f1)
            print(['({0}: {1})'.format(k, sia[k]) for k in sia])    
            wordcloud = WordCloud().generate(f1)
            fig, axes= plt.subplots(figsize=(20,12),clear=True)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.show()
            wordcloud.to_file("wordcloud_images/img.png")

             #From the Vader Library, we will perform both sentiment and intensity analysis
        else:
            f1=input("Enter the text for performing Text Analytics: ")
            sia = senti.polarity_scores(f1)
            print(['({0}: {1})'.format(k, sia[k]) for k in sia])
            wordcloud = WordCloud().generate(f1)
            fig, axes= plt.subplots(figsize=(20,12),clear=True)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.show()
            wordcloud.to_file("wordcloud_images/img.png")
    except Exception as e: # work on python 3.x
        print(str(e))


# In[87]:


def math(match):
    s=int(match.groups()[0])
    s1=int(match.groups()[1])
    input("What operation would you ")
    print(type(match))
    return s+s1


# In[174]:


c=['ae' ,'ar' ,'at' ,'au' ,'be' ,'bg' ,'br' ,'ca' ,'ch' ,'cn' ,'co' ,'cu' ,'cz' ,'de' ,'eg' ,'fr' ,'gb' ,'gr' ,'hk' ,'hu' ,'id' ,'ie' ,'il' ,'in' ,'it' ,'jp' ,'kr' ,'lt' ,'lv' ,'ma' ,'mx' ,'my' ,'ng' ,'nl' ,'no' ,'nz' ,'ph' ,'pl' ,'pt' ,'ro' ,'rs' ,'ru' ,'sa' ,'se' ,'sg' ,'si' ,'sk' ,'th' ,'tr' ,'tw' ,'ua' ,'us' ,'ve' ,'za']
cat=['business', 'entertainment', 'general health' ,'science', 'sports', 'technology']
#category='busine'
#country='usa'
cat_miss=[]
country_miss=[]
if category in cat and country in c:   
    top_headlines = newsapi.get_top_headlines(category=category,language='en',country=country)
    top_headlines=json_normalize(top_headlines['articles'])
    top_headlines=top_headlines.head()
    top_headlines['country']=i
    top_headlines['category']=j
    print(top_headlines.head(1))
elif category not in cat and country in c:
    for j in cat:
        if (fuzz.partial_ratio(category,j)>50):
            cat_miss.append(j)
    d=set(cat_miss)
    print(str(d))
elif country not in c and category in cat:
    for i in c:
        if (fuzz.partial_ratio(country,i)>70):
            country_miss.append(i)
    d1=set(country_miss)
    print(str(d1))
else:
    if len(d)==1:
        print("Could not find that category, did you mean this?"+str(d))
    else:
        print("Could not find that category, did you mean any of these?"+str(d))       
    if len(d1)==1:
        print("Could not find that country, did you mean this?"+str(d1))
    else:
        print("Could not find that country, did you mean any of these?"+str(d1))
    


# In[178]:


def top_headlines(match):
    #country=input("Which country are you interested in?")
    try:
        category=match.groups()[1]
        
        country=match.groups()[0]
        c=['ae' ,'ar' ,'at' ,'au' ,'be' ,'bg' ,'br' ,'ca' ,'ch' ,'cn' ,'co' ,'cu' ,'cz' ,'de' ,'eg' ,'fr' ,'gb' ,'gr' ,'hk' ,'hu' ,'id' ,'ie' ,'il' ,'in' ,'it' ,'jp' ,'kr' ,'lt' ,'lv' ,'ma' ,'mx' ,'my' ,'ng' ,'nl' ,'no' ,'nz' ,'ph' ,'pl' ,'pt' ,'ro' ,'rs' ,'ru' ,'sa' ,'se' ,'sg' ,'si' ,'sk' ,'th' ,'tr' ,'tw' ,'ua' ,'us' ,'ve' ,'za']
        cat=['business', 'entertainment', 'general health' ,'science', 'sports', 'technology']
        #category='busine'
        #country='usa'
        cat_miss=[]
        country_miss=[]
        if category in cat and country in c:   
            top_headlines = newsapi.get_top_headlines(category=category,language='en',country=country)
            top_headlines=json_normalize(top_headlines['articles'])
            top_headlines=top_headlines.head()
            top_headlines['country']=i
            top_headlines['category']=j
            top_headlines = newsapi.get_top_headlines(category=cat,language='en',country=c)
            top_headlines=json_normalize(top_headlines['articles'])
            top_headlines=top_headlines.head()
            newdf=top_headlines[["title","url"]]
            dic=newdf.set_index('title')['url'].to_dict()
            engine.say("Here are some of the top articles")
            engine.runAndWait()
            for (k,v) in dic.items():
                engine.say(k)
                engine.say('You can find more info here:')
                engine.runAndWait()
                engine.stop()
                print(k+"\n\n"+v)
        elif category not in cat and country in c:
            for j in cat:
                if (fuzz.partial_ratio(category,j)>50):
                    cat_miss.append(j)
            d=set(cat_miss)
            if len(d)==1:
                print("Could not find that category, did you mean this?"+str(d))
            else:
                print("Could not find that category, did you mean any of these?"+str(d))    
        elif country not in c and category in cat:
            for i in c:
                if (fuzz.partial_ratio(country,i)>70):
                    country_miss.append(i)
            d1=set(country_miss)
            if len(d1)==1:
                print("Could not find that country, did you mean this?"+str(d1))
            else:
                print("Could not find that country, did you mean any of these?"+str(d1))
        else:
            if len(d)==1:
                print("Could not find that category, did you mean this?"+str(d))
            else:
                print("Could not find that category, did you mean any of these?"+str(d))       
            if len(d1)==1:
                print("Could not find that country, did you mean this?"+str(d1))
            else:
                print("Could not find that country, did you mean any of these?"+str(d1))
    except Exception as e: # work on python 3.x
        engine.say(str(e))
        


# In[179]:


pairs = [
    ["(Hi|Hello|Hey) I'm (.*)", ["""Hey %1, What can I do for you? You can ask me for Stocks information, 
    ask me to generate wordclouds or for the latest news
                                 """], None],
   ["number (.*)+(.*)",["nothing"],math],
#    Enter stock symbol for the below 3 cases: To test it out, you can type in the chat "open Apple stock" or "open AAPL stock"
    # you can type both stock symbol or company name.
    ["open (.*) live Stock", [""], stock],
    # for this below case, type "open Apple yearly" to get stock trend for the current year for Apple
    ["open (.*) Yearly", ["Yearly stock trend for " + "%1".upper()], yearly_stock_trend_complete],
    ["compare (.*), (.*), (.*), (.*), (.*)", ["Yearly stock trend for " + "%1".upper()+", %2".upper()+", %3".upper()+", %4".upper()+", %5".upper()], yearly_stock_compare],
   
    ["open (.*) 5", ["Five Yearly stock trend for " + "%1".upper()], five_yearly_stock_trend_complete],
 # for this below case, type "open Apple monthly" to get stock trend for the current month for Apple
    ["open (.*) Monthly", ["Monthy stock trend for " + "%1".upper()], monthly_stock_trend_complete],
# Enter Text to perform Text Analytics. First type text type: document/text. If document enter document path else enter text 
    ["perform TA of (.*)", [""], sentiment],
    ["generate wordclouds of (.*)", [""], sentiment],
    
    #["news (.*)",["Here's the latest news \n"], general_headlines],
 #Get the top headlines: enter country followed by category   
    ["(.*) (.*) news", [""], top_headlines],
     ["open (.*) dividends", [""], dividends],
    ["open (.*) actions", [""], actions],
      ["open (.*) sustainability", [""], sustainability],
    ["open (.*) recommendations", [""], recommendations],
    ["open (.*) analyst info", [""], analyst_info],
  
  ["open (.*) options", [""], options],
  ["open (.*) isin", [""], isin],
  
    ["open (.*) splits", [""], splits],
    ["open (.*) financials", [""], financials],
    ["open (.*) quarterly financials", [""], quarterly_financials],
["open (.*) balance sheet", [""], balance_sheet],
    ["open (.*) quarterly balance sheet", [""], quarterly_balance_sheet],
["open (.*) earnings", [""], earnings],
    ["open (.*) quarterly earnings", [""], quarterly_earnings],
    ["open (.*) cashflow", [""], cashflow],
    ["open (.*) quarterly cashflow", [""], cashflow],
["open (.*) calendar", [""], calendar],

 ["open (.*) gainers", [""], day_gainers],
 ["open (.*) losers", [""], day_losers],
 ["open (.*) stats", [""], day_stats],
 ["open (.*) stats valuation", [""], day_stats_valuation],
     ["open (.*) day most active", [""], get_day_most_active],
 ["open (.*) losers", [""], day_losers],
    ["open (.*) major holders", [""], major_holders],
    ["open (.*) institutional holders", [""], institutional_holders],
]


# In[180]:


def chat():
    print("Greetings! My name is Muku, What is yours?.")
    Chatbot = Muku_Chat(pairs, reflections)
    Chatbot.converse()
if __name__=="__main__":
    chat()


# def voice_input():
#     global a2t
#     import speech_recognition as sr
#     r = sr.Recognizer()
#     with sr.Microphone() as source:
#         #r.adjust_for_ambient_noise(source)
#         r.record(source,duration=2)
#         print("Say something!")
#         audio = r.listen(source)
#     try:
#         a2t=r.recognize_sphinx(audio,keyword_entries=[('forward',1.0),('backward',1.0),('left',1.0),('right',1.0),('stop',1.0),('find line',0.95),('follow',1),('lights on',1),('lights off',1)])
#         print("Sphinx thinks you said " + a2t)
#     except sr.UnknownValueError:
#         print("Sphinx could not understand audio")
#     except sr.RequestError as e:
#         print("Sphinx error; {0}".format(e))
#     BtnVIN.config(fg=color_text,bg=color_btn)
#     return a2t 
# voice_input()

# from __future__ import print_function
# import pickle
# import os.path
# from googleapiclient.discovery import build
# from google_auth_oauthlib.flow import InstalledAppFlow
# from google.auth.transport.requests import Request
# 
# # If modifying these scopes, delete the file token.pickle.
# SCOPES = ['https://www.googleapis.com/auth/gmail.readonly','https://www.googleapis.com/auth/gmail.labels'
#           ,'https://www.googleapis.com/auth/gmail.send','https://www.googleapis.com/auth/gmail.compose']
# 
# def main():
#     """Shows basic usage of the Gmail API.
#     Lists the user's Gmail labels.
#     """
#     creds = None
#     # The file token.pickle stores the user's access and refresh tokens, and is
#     # created automatically when the authorization flow completes for the first
#     # time.
#     if os.path.exists('token.pickle'):
#         with open('token.pickle', 'rb') as token:
#             creds = pickle.load(token)
#     # If there are no (valid) credentials available, let the user log in.
#     if not creds or not creds.valid:
#         if creds and creds.expired and creds.refresh_token:
#             creds.refresh(Request())
#         else:
#             flow = InstalledAppFlow.from_client_secrets_file(
#                 'conversational-app-38c04-45b7b0ce9e97.json', SCOPES)
#             creds = flow.run_local_server(port=0)
#         # Save the credentials for the next run
#         with open('token.pickle', 'wb') as token:
#             pickle.dump(creds, token)
# 
#     service = build('gmail', 'v1', credentials=creds)
# 
#     # Call the Gmail API
#     results = service.users().labels().list(userId='mukundan.sankar14@gmail.com').execute()
#     labels = results.get('labels', [])
# 
#     if not labels:
#         print('No labels found.')
#     else:
#         print('Labels:')
#         for label in labels:
#             print(label['name'])
# 
# if __name__ == '__main__':
#     main()

# from __future__ import print_function
# from googleapiclient.discovery import build
# from apiclient import errors
# from httplib2 import Http
# from email.mime.text import MIMEText
# import base64
# from google.oauth2 import service_account
# 
# # Email variables. Modify this!
# EMAIL_FROM = 'noreply@lyfpedia.com'
# EMAIL_TO = 'mark.zuckerber@facebook.com'
# EMAIL_SUBJECT = 'Hello  from Lyfepedia!'
# EMAIL_CONTENT = 'Hello, this is a test\nLyfepedia\nhttps://lyfepedia.com'
# 
# service = service_account_login()
# # Call the Gmail API
# message = create_message(EMAIL_FROM, EMAIL_TO, EMAIL_SUBJECT, EMAIL_CONTENT)
# sent = send_message(service,'me', message)
# 
# 

# def create_message(sender, to, subject, message_text):
#   """Create a message for an email.
#   Args:
#     sender: Email address of the sender.
#     to: Email address of the receiver.
#     subject: The subject of the email message.
#     message_text: The text of the email message.
#   Returns:
#     An object containing a base64url encoded email object.
#   """
#   message = MIMEText(message_text)
#   message['to'] = to
#   message['from'] = sender
#   message['subject'] = subject
#   return {'raw': base64.urlsafe_b64encode(message.as_string())}
# 
# def send_message(service, user_id, message):
#   """Send an email message.
#   Args:
#     service: Authorized Gmail API service instance.
#     user_id: User's email address. The special value "me"
#     can be used to indicate the authenticated user.
#     message: Message to be sent.
#   Returns:
#     Sent Message.
#   """
#   try:
#     message = (service.users().messages().send(userId=user_id, body=message)
#                .execute())
#     print('Message Id: %s' % message['id'])
#     return message
#   except errors.HttpError as error:
#     print('An error occurred: %s' % error)
# 
# def service_account_login():
#   SCOPES = ['https://www.googleapis.com/auth/gmail.send']
#   SERVICE_ACCOUNT_FILE = 'service-key.json'
# 
#   credentials = service_account.Credentials.from_service_account_file(
#           SERVICE_ACCOUNT_FILE, scopes=SCOPES)
#   delegated_credentials = credentials.with_subject(EMAIL_FROM)
#   service = build('gmail', 'v1', credentials=delegated_credentials)
#   return service
