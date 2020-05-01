#!/usr/bin/env python
# coding: utf-8

# # Problem Statement: Building a basic NLP based chatbot 

# In[34]:


# import stock_info module from yahoo_fin
from yahoo_fin import stock_info as si
from flask import Flask, render_template, request
import plotly.express as px
import seaborn as sns


# In[35]:


from nltk.chat.util import Chat, reflections
import datetime
now=datetime.datetime.now()
import yfinance as yf
msft = yf.Ticker("MSFT")
import pandas as pd
import os
os.getcwd()
import re
import msvcrt
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.express as px


# In[36]:


from get_all_tickers import get_tickers as gt

list_of_tickers = gt.get_tickers()
# or if you want to save them to a CSV file
n=list_of_tickers


# In[37]:


def yearly_stock_trend_complete(s):
    d=pd.DataFrame()
    if s.upper() in n:
        d=d.append(yf.Ticker(s).history(period='1y',interval='1wk').reset_index())
    fig = px.line(d, x="Date", y="Close",
                  labels={'Close':'Closing Stock Price'}, 
                  template='plotly_dark',
                 color_discrete_sequence=[ "aqua"],
                  title="Closing Stock Price for the Current Year for "+str(s).upper()
                 )
    return fig.show()


# In[38]:



def monthly_stock_trend_complete(s):
    d=pd.DataFrame()
    if s.upper() in n:
        d=d.append(yf.Ticker(s).history(period='1mo',interval='1d').reset_index())
    fig = px.line(d, x="Date", y="Close",
                  labels={'Close':'Closing Stock Price'}, 
                  template='plotly_dark',
                 color_discrete_sequence=[ "aqua"],
                  title="Closing Stock Price for the Current Month for "+str(s).upper()
                 )
    return fig.show()


# In[39]:


def stock(s):
    d=''
    if s in n:
        d+=s
    return "Current stock price of "+str(s).upper()+" is: "+ str(si.get_live_price(s))


# In[40]:


reflections = {
    "i am": "you are",
    "i was": "you were",
    "i": "you",
    "i'm": "you are",
    "i'd": "you would",
    "i've": "you have",
    "i'll": "you will",
    "my": "your",
    "you are": "I am",
    "you were": "I was",
    "you've": "I have",
    "you'll": "I will",
    "your": "my",
    "yours": "mine",
    "you": "me",
    "me": "you",
}


# In[41]:


pairs=[   
    
        [
        r"my name is (.*)",
        ["Hello %1, How are you today ?",]
    ],
     [
        r"what is your name ?",
        ["My name is Mukundan and I'm a chatbot ?",]
    ],
    [
        r"how are you ?",
        ["I'm doing good\nHow about You ?",]
    ],
    [
        r"sorry (.*)",
        ["Its alright %1 Its OK, never mind",]
    ],
    
   
    [
        r"i'm (.*) doing good",
        ["Nice to hear that","Alright :)",]
    ],
    [
        r"hi|hey|hello",
        ["Hello", "Hey there",]
    ],
  [
        r"(.*) time?".lower(),
        ["The time is "+now.strftime("%H:%M:%S"),]
        
    ],
 

[         r"quit|QUIT|Quit|Exit|exit|bye|Bye",
        ["BBye take care. See you soon :) ","It was nice talking to you. See you soon :)"]
],

]


# In[42]:


module_list=['stock','text analytics']


# In[43]:


from sent_analy import sentiment


# In[44]:


import pprint
pp = pprint.PrettyPrinter(indent=4)
def sentiment_d(text):
    pp.pprint(text);


# In[45]:


import wordcloud
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


# In[46]:


def sentiment_d(txt):
   
    print(sentiment(txt));

    wordcloud = WordCloud().generate(txt)
    fig, axes= plt.subplots(figsize=(20,12),clear=True)

    plt.imshow(wordcloud, interpolation='bilinear')
  
    plt.show()


# In[61]:


def getStockPrice(txt):
    if txt.upper() in n and txt!='quit stock':
        print(stock(txt));
        print(yearly_stock_trend_complete(txt))
        print(monthly_stock_trend_complete(txt))


# In[62]:


def chatbot():
    txt = input("""Enter the module: stock/text analytics: (Type quit to end our conversation): \n\n""")

    if txt in module_list:
        
        while txt.lower()=="STOCK".lower() and txt!='quit stock':
            txt=input("Welcome to stock app. Please enter the stock symbol of the company: (Type quit to leave stock app)\n\n")
            while txt.upper() in n and txt!='quit stock':
                print(getStockPrice(txt))
                txt = input("""Continue entering stock symbol. (Type quit to leave stock app) \n\n""")
            txt = input("""Enter the module: stock/text analytics: (Type quit to enter a normal conversation with me) \n\n""")
            
        while txt.lower()=="text analytics".lower() and txt!='quit TA':
            txt=input("Welcome to the Text Analysis App. Please enter the text you would like to see an analysis on. (Type quit to leave text analysis app) \n\n")
            while len(txt)>=2 and txt!='quit TA':
                print(sentiment_d(txt))
                txt = input("""Continue entering text. (Type quit to leave text analysis app) \n\n""")
            txt = input("""Enter the module: stock/text analytics: (Type quit to enter a normal conversation with me): \n\n""")
    else:
        print("Entering general chat \n\n")
    chat = Chat(pairs, reflections)
    chat.converse()
    


# In[63]:


if __name__ == "__main__":
    chatbot()


# In[ ]:




