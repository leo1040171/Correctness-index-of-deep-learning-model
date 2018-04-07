import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import textblob
from textblob import TextBlob
#data = "Hello! My Name is Muhammad Ali. I love programming!. I like music. I hate cooking. I am the worst Programmer ever!"
#Creating Object
#tblob = TextBlob(data)
#to get sentences from above data
#tblob.sentences

#analysing each word as NN, VRZ, PRP etc
#tags = tblob.tags
#iterating through each sentence
f#or sen in tblob.sentences:
  #  print(sen)
   # print(sen.polarity)
    #print(sen.subjectivity)
    
def percentage(count, NoOfTerms):
    return 100*float(count)/int(NoOfTerms)    
    
import tweepy
import sys

consumerKey = '-----------------'
consumerSecret = '------------------------'
accessToken = '-----------------------'
accessTokenSecret = '-----------------------'

auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
auth.set_access_token(accessToken, accessTokenSecret)
api = tweepy.API(auth)

searchTerm = input("Enter Keyword/Tag to search about: ")
NoOfTerms = int(input("Enter how many tweets to search: "))

tweets = tweepy.Cursor(api.search, q=searchTerm).items(NoOfTerms)

neutral = 0
positive = 0
strongly_positive = 0
negative = 0
strongly_negative = 0


for tweet in tweets:
    
    analysis = TextBlob(tweet.text)
    #print(tweet.text)
    #print(analysis.sentiment.polarity)
    if (analysis.sentiment.polarity == 0):
        neutral += 1
    elif (analysis.sentiment.polarity > 0.00 and analysis.sentiment.polarity < 0.50 ):
        positive += 1
    elif (analysis.sentiment.polarity >= 0.50 and analysis.sentiment.polarity <= 1.00 ):
        strongly_positive += 1
    elif (analysis.sentiment.polarity < 0.00 and analysis.sentiment.polarity> -0.50):
        negative += 1
    elif (analysis.sentiment.polarity <= -0.50 and analysis.sentiment.polarity> -1.00):
        strongly_negative += 1
        
prct_neutral = percentage(neutral,NoOfTerms)
prct_positive = percentage(positive,NoOfTerms)
prct_negative = percentage(negative,NoOfTerms)
prct_strongly_positive = percentage(strongly_positive,NoOfTerms)
prct_strongly_negative = percentage(strongly_negative,NoOfTerms) 

neutral = format(prct_neutral, '.2f')
positive = format(prct_positive, '.2f') 
negative = format(prct_negative, '.2f')
strongly_positive = format(prct_strongly_positive, '.2f') 
strongly_negative = format(prct_strongly_negative, '.2f')    

labels = ['positive =' + str(prct_positive) + '%','strongly_positive =' + str(prct_strongly_positive) + '%','negative =' + str(prct_negative) + '%','strongly_negative =' + str(prct_strongly_negative) + '%','neutral =' + str(prct_neutral) + '%' ]     
sizes = [positive,strongly_positive,negative,strongly_negative,neutral]
colors = ['black','grey','blue','yellow','green']
patches, texts = plt.pie(sizes,colors = colors, startangle = 90)
plt.legend(patches,labels,loc ="best")
plt.title('How people are reacting on ' + searchTerm + ' by analysing ' + str(NoOfTerms) +' latest tweets')
plt.axis('equal')
plt.tight_layout()
plt.show()
