import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import datetime
from vega_datasets import data
import geopandas as gpd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize 

import string
import nltk
from nltk import PorterStemmer
from nltk import WordNetLemmatizer
import re

nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('wordnet')
stopwords=nltk.corpus.stopwords.words('english')


st.title('Fake News Prediction')
st.header('Understanding Fake News and How Models Distinguish Them')

intro_text="With the current massive use of social media, information quality becomes an important issue when it comes to news reporting and other sources of factuality checking. Given the hot topics in fake news classification and toxic comments classification, many datasets have been collected, and they cover a wide range of domains. For example, the LIAR dataset was collected from Politifact and covers the political topics, while FEVER which covers scientific topics is collected from Wikipedia. As more and more fake news prevails on social media, it is getting harder for people to distinguish between true news and fake news and people could sometimes easily be misled by the seemingly-true fake news. As a result, different fake news prediction models have developed intending to serve as fake news detector and help people perform fact-checking. Many state-of-the-art fake news prediction models have achieved satisfying performance and even outperforming human. This article aims to investigate the characteristics of fake news articles, focusing on a selected fake news dataset, and studying how fake news prediction models distinguish between true news and fake news. This allows people to understand how those models differentiate true news and fake news and would also be beneficial for improving reader's ability in identifying fake news. "

st.write(intro_text)


st.header('On what subjects do people tend to lie?')



columns=['id','label','statement','subject','speaker', 'job', 'state','party','barely_true_counts','false_counts',
                  'half_true_counts','mostly_true_counts','pants_on_fire_counts','context']
label_values=['false','pants-fire','barely-true','true','mostly-true','false','half-true']
meta_feature=['subject','speaker', 'job', 'state','party']

def read_data():
    
    df_train=pd.read_csv('liar_dataset/train.tsv', delimiter='\t', header=None, names=columns)
    df_test=pd.read_csv('liar_dataset/test.tsv', delimiter='\t', header=None, names=columns)
    df_valid=pd.read_csv('liar_dataset/valid.tsv', delimiter='\t', header=None, names=columns)
    df_total=pd.concat([df_train, df_test, df_valid]).reset_index(drop=True)
    return df_train, df_test, df_valid, df_total

df_train, df_test, df_valid, df_total=read_data()

st.write('Example news statement from LIAR dataset')
df_train.statement[:10]

combine_labels=st.checkbox('Combined labels')

top_n=st.slider(
    'Select the number of entries to show',
     1, 20)

# label_sel=st.selectbox(
#     'Select a label value: ',
#      label_values)

kind=['subject','speaker', 'job', 'state']
def meta_feature_filtering(df, top_n, label, feature_sel):
    df_sub=df_train.loc[df_train.label==label]
    sel=pd.DataFrame(df_sub[feature_sel].value_counts()[:top_n]).reset_index()
    sel.columns=['kind','count']
    return sel

def meta_feature_filtering_combined(df, top_n, kind):
    total=[]
    
    for l in label_values:
        df_sub=df_train.loc[df_train.label==l]
        sel=pd.DataFrame(df_sub[kind].value_counts()[:top_n]).reset_index()
        sel.columns=['kind',f'{l}']
        row_name=sel.iloc[:,0]
        total.append(sel.iloc[:,1])
    total.append(row_name)
    return pd.concat(total, axis=1)

display_type=['Absolute', 'Percentage']

absolute=st.selectbox(
        'Select type of values: ',
        display_type)

feature_sel=st.selectbox(
    'Select a Meta Feature: ',
     meta_feature)

combined_table=meta_feature_filtering_combined(df_train, top_n, feature_sel)
if absolute=='Percentage':
    v=combined_table.iloc[:,-1]
    combined_table['sum']=combined_table.sum(axis=1)
    combined_table.iloc[:, :-2]=combined_table.iloc[:, :-2].div(combined_table['sum'], axis=0)
    combined_table=combined_table.drop('sum', axis=1)
combined_table=combined_table.melt(id_vars='kind')

if combine_labels:
    
    scatter_chart=st.altair_chart(
        alt.Chart(combined_table, width=700).mark_bar().encode(
            x='value:Q',
            y=alt.Y('kind:N', sort='-x'),
            color='variable:N'
        ).interactive()
    )

else:

    label_sel=st.selectbox(
        'Select a label value: ',
         label_values)

    # combined_table=meta_feature_filtering(df_train, top_n, label_sel, feature_sel)
    combined_table=combined_table.loc[combined_table.variable==label_sel]
    combined_table=combined_table.drop('variable', axis=1)

    scatter_chart=st.altair_chart(
        alt.Chart(combined_table).mark_bar().encode(
            x='value:Q',
            y=alt.Y('kind:N', sort='-x')
        ).interactive()
    )



st.header('Word Frequency and use of punctuation')


def text_lowercase(text): 
    return text.lower() 

def remove_punctuation(text): 
    translator=str.maketrans('', '', string.punctuation) 
    return text.translate(translator) 

def remove_whitespace(text): 
    return  " ".join(text.split()) 

def remove_stopwords(text): 
    word_tokens=word_tokenize(text) 
    filtered_text=[word for word in word_tokens if word not in stopwords] 
    return " ".join(filtered_text)

def stem_words(text): 
    word_tokens=word_tokenize(text) 
    stems=[stemmer.stem(word) for word in word_tokens] 
    return " ".join(stems) 

lemmatizer=WordNetLemmatizer() 
stemmer=PorterStemmer()

def lemmatize_word(text): 
    word_tokens=word_tokenize(text) 
    lemmas=[lemmatizer.lemmatize(word) for word in word_tokens] 
    return " ".join(lemmas) 

def preprocess_statement(df):
    df['new_statement']=df.statement.apply(lambda x: text_lowercase(x))
    df['new_statement']=df.new_statement.apply(lambda x: remove_punctuation(x))
    df['new_statement']=df.new_statement.apply(lambda x: remove_whitespace(x))
    df['new_statement']=df.new_statement.apply(lambda x: remove_stopwords(x))
    df['new_statement']=df.new_statement.apply(lambda x: stem_words(x))
    df['new_statement']=df.new_statement.apply(lambda x: lemmatize_word(x))
    return df

def extract_key_words(df, label, subject):
    sub_df=df.loc[df.label==label]
    sub_df=sub_df.loc[sub_df.subject==subject]
    total_text=sub_df.new_statement.sum()
    return total_text

top_ten_subjects=df_train.subject.value_counts()[:10].index

df_train=preprocess_statement(df_train)

label_sel=st.selectbox(
        'Select a label value: ',
         label_values, key='label_sel2')

subject_type=st.selectbox(
        'Select a subject topic: ',
         top_ten_subjects)

total_text=extract_key_words(df_train, label_sel, subject_type)

fig=plt.figure(figsize=(10, 5))
wordcloud=WordCloud(stopwords=stopwords, background_color="white").generate(total_text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
st.pyplot(fig)

st.header('Sentiment analysis on true news and fake news')


sid=SentimentIntensityAnalyzer()


def analyze_sentiment(sentence):
    score=sid.polarity_scores(sentence)
#     for k in sorted(score):
#         print('{0}: {1}, '.format(k, score[k]), end='')
#     print()
    return score

def sentiment_for_news(df, label):
    sub_df=df.loc[df.label==label].reset_index(drop=True)
    scores={}
    for i in range(0, len(sub_df)):
        score=analyze_sentiment(sub_df.statement[i])
        scores['compound']=scores.get('compound',[])+[score['compound']]
        scores['neg']=scores.get('neg',[])+[score['neg']]
        scores['neu']=scores.get('neu',[])+[score['neu']]
        scores['pos']=scores.get('pos',[])+[score['pos']]
    return pd.DataFrame.from_dict(scores)

def sentiment_for_type(df, sentiment_type):
    scores={}
    for v in label_values:
        sub_df=df.loc[df.label==v].reset_index(drop=True)
        for i in range(0, len(sub_df)):
            score=analyze_sentiment(sub_df.statement[i])
            scores[v]=scores.get(v,[])+[score[sentiment_type]]
        
    return scores

sentiment_type=['compound','neg','neu','pos']
sentiment_type_table=sentiment_for_news(df_train, 'true')
sentiment_type_table

alt.Chart(sentiment_type_table).transform_fold(
    ['compound',
     'neg',
     'neu',
     'pos'],
    as_ = ['Sentiments', 'value']
).transform_density(
    density='value',
    bandwidth=0.3,
    groupby=['Sentiments'],
    extent= [-2, 5]
).mark_area().encode(
    alt.X('value:Q'),
    alt.Y('density:Q', axis=None),
    alt.Row('Sentiments:N')
    
).properties(width=400, height=50)