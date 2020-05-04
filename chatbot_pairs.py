#!/usr/bin/env python
# coding: utf-8

# In[22]:


import nltk
from nltk.chat.util import Chat, reflections
import re
import random
import webbrowser
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
from pandas.io.json import json_normalize
pd.set_option('display.max_colwidth', -1)


# In[23]:


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
            except EOFError:
                print(user_input)
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


# In[103]:


def top_headlines(match):
    #country=input("Which country are you interested in?")
    category=match.groups()[1]
    country=match.groups()[0]
    
    #category=input("Which category are you interested in?")
    #input1=print("Enter the country, category followed by news: \n\n example: us business news")
    top_headlines = newsapi.get_top_headlines(category=category,language='en',country=country)
    top_headlines=json_normalize(top_headlines['articles'])
    newdf=top_headlines[["title","url"]]
    dic=newdf.set_index('title')['url'].to_dict()
    top_headlines
    print("Here are some of the top articles\n\n")
    for (k,v) in dic.items():
        print(k+"\n\n"+v)
        #urn (top_headlines.url(10),top_headlines['title'].head(10),top_headlines['description'].head(10),top_headlines['content'].head(10))
#top_headlines()


# list_of_tickers = gt.get_tickers()
# n=list_of_tickers
# 

# In[104]:


table=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
df=table[0]
newdf=df[["Symbol","Security"]]
dic=newdf.set_index('Symbol')['Security'].to_dict()


# In[105]:


pattern = re.compile("M Company")
#pattern.search([i for i in dic.keys])    
# Match at index 0
s='MMM'
k='MMM'
re.compile(s.lower()).match(k.lower())


# In[106]:


def stock(match):
    s=match.groups()[0]
    print(s)
    for (k,v) in dic.items():
        if re.compile(s.lower()).match(v.lower()):
            print("Current stock price of "+str(v).upper()+" is: "+ str(np.round(si.get_live_price(k),2)))
        elif re.compile(s.lower()).match(k.lower()):
            print("Current stock price of "+str(v).upper()+" is: "+ str(np.round(si.get_live_price(k),2)))
       


# In[107]:


def yearly_stock_trend_complete(match):
    s=match.groups()[0]
    d=pd.DataFrame()
    for (k,v) in dic.items():
        if re.compile(s.lower()).match(v.lower()):
            d=d.append(yf.Ticker(k).history(period='1y',interval='1wk').reset_index())
        elif re.compile(s.lower()).match(k.lower()):
            d=d.append(yf.Ticker(k).history(period='1y',interval='1wk').reset_index())
    fig = px.line(d, x="Date", y="Close",
                  labels={'Close':'Closing Stock Price'}, 
                  template='plotly_dark',
                 color_discrete_sequence=[ "aqua"],
                  title="Closing Stock Price for the Current Year for "+str(s)
                 )
    fig.write_image("images/"+ str(s) +".png")    
    fig.write_image("images/"+ str(s) +".png")    
    return fig.show()


# In[108]:


def monthly_stock_trend_complete(match):
    s=match.groups()[0]
    d=pd.DataFrame()
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
    

    #img_bytes = fig.to_image(format="png")
    #from IPython.display import Image
    
    #return Image(img_bytes)


# In[109]:


def sentiment_d(match):
    s=match.groups()[0]
    wordcloud = WordCloud().generate(s)
    fig, axes= plt.subplots(figsize=(20,12),clear=True)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.show()
    wordcloud.to_file("wordcloud_images/"+ str(s[0:7]) +".png")

    


# In[110]:


def sentiment(match):
    txt=match.groups()[0]
    senti = SentimentIntensityAnalyzer()
    if (txt.upper()=="document".upper()) or (txt.upper()=="file".upper()):
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


# In[113]:


pairs = [
    ["Hi im (.*)", ["hello %1, What can I do for you?"], None],
#    Enter stock symbol for the below 3 cases: To test it out, you can type in the chat "open Apple stock" or "open AAPL stock"
    # you can type both stock symbol or company name.
    ["open (.*) Stock", [""], stock],
    # for this below case, type "open Apple yearly" to get stock trend for the current year for Apple
    ["open (.*) Yearly", ["Yearly stock trend for " + "%1".upper()], yearly_stock_trend_complete],
 # for this below case, type "open Apple monthly" to get stock trend for the current month for Apple
    ["open (.*) Monthly", ["Monthy stock trend for " + "%1".upper()], monthly_stock_trend_complete],
# Enter Text to perform Text Analytics. First type text type: document/text. If document enter document path else enter text 
    ["perform TA of (.*)", [""], sentiment],
    ["news (.*)",["Here's the latest news \n"], general_headlines],
 #Get the top headlines: enter country followed by category   
    ["(.*) (.*) news", [""], top_headlines],
 
]


# In[114]:


def chat():
    print("Greetings! My name is Chatbot-T1, What is yours?.")
    Chatbot = Muku_Chat(pairs, reflections)
    Chatbot.converse()
if __name__=="__main__":
    chat()


# In[ ]:




