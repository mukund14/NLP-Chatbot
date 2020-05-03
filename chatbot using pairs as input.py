#!/usr/bin/env python
# coding: utf-8

# In[28]:


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


# In[29]:


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


# In[30]:


list_of_tickers = gt.get_tickers()
n=list_of_tickers


# In[31]:


table=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
df=table[0]
newdf=df[["Symbol","Security"]]
dic=newdf.set_index('Symbol')['Security'].to_dict()


# In[32]:


pattern = re.compile("M Company")
#pattern.search([i for i in dic.keys])    
# Match at index 0
s='MMM'
k='MMM'
re.compile(s.lower()).match(k.lower())


# In[33]:


def stock(match):
    s=match.groups()[0]
    for (k,v) in dic.items():
        if re.compile(s.lower()).match(v.lower()):
            print("Current stock price of "+str(v).upper()+" is: "+ str(np.round(si.get_live_price(k),2)))
        elif re.compile(s.lower()).match(k.lower()):
            print("Current stock price of "+str(v).upper()+" is: "+ str(np.round(si.get_live_price(k),2)))
       


# In[55]:


def sentiment_d(match):
    s=match.groups()[0]
    print(sentiment(s));
    wordcloud = WordCloud().generate(s)
    fig, axes= plt.subplots(figsize=(20,12),clear=True)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.show()


# In[56]:


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
    return fig.show()


# In[57]:


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
    return fig.show()


# In[58]:


pairs = [
    ["Hi im (.*)", ["hello %1, What can I do for you?"], None],
#    ["Open google", ["opening www.google.com"], open_google],
    ["open (.*) Stock", [""], stock],
    ["open (.*) Yearly", ["Yearly stock trend for " + "%1".upper()], yearly_stock_trend_complete],
    ["open (.*) Monthly ", ["Monthy stock trend for %1.upper()"], monthly_stock_trend_complete],
    ["perform TA of (.*)", [""], sentiment_d],
  #  ["Open (.*) Stock", [""], stock],
]


# In[59]:


def chat():
    print("Greetings! My name is Chatbot-T1, What is yours?.")
    Chatbot = Muku_Chat(pairs, reflections)
    Chatbot.converse()
if __name__=="__main__":
    chat()


# app = Flask(__name__, template_folder='templates')
# 
# @app.route('/', methods=['GET', 'POST'])
# def samplefunction():
#     if request.method == 'GET':
#         return render_template('new.html')
#     if request.method == 'POST':
#         greetIn = request.form['human']
#         greetOut = chat()
#         return render_template('new.html',bot1=greetOut)
# if __name__ == '__main__':
#     app.run()
#     

# import os
# app = Flask(__name__)
# #app.config["DEBUG"] = True
# @app.route("/" ,methods=['GET'])
# def home():    
#     return render_template('index.html') 
# @app.route("/" ,methods=['POST'])
# def get_bot_response():    
#     userText = request.form.get('stock')
#    
#     
#     return showResult(userText)
# 
# @app.route("/" ,methods=['GET'])
# def showResult(inputVal): 
#     chatVal = chat()
#    # print(chat_var)
#     return render_template('result.html',val=chatVal);
# 
#    
# if __name__ == "__main__":    
#     app.run()
# 
