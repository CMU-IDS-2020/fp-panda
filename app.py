import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
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
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.corpus.reader.wordnet import WordNetError

nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
stopwords=nltk.corpus.stopwords.words('english')


st.title("Learn to Predict Fake News")
st.header("Let's start with a small test")
st.subheader("Could you correctly predict whether the following news statements are true or fake?")

st.write("Says 57 percent of federal spending goes to the military and just 1 percent goes to food and agriculture, including food stamps.	federal-budget,military,poverty")
st.write("Topic: federal-budget,military,poverty")
st.write("Speaker: facebook-posts")
st.write("Job: Social media posting")
option1 = st.selectbox(
    'True or False',
     ["I don't know", 'True', 'False'], key="option1")

'You think the statement is:', option1


st.write("The Fed created $1.2 trillion out of nothing, gave it to banks, and some of them foreign banks, so that they could stabilize their operations.")
st.write("Topic: economy,financial-regulation")
st.write("Speaker: dennis-kucinich")
st.write("Job: U.S. representative")
st.write("Party: democrat")
option2 = st.selectbox(
    'True or False',
     ["I don't know", 'True', 'False'], key="option2")

'You think the statement is:', option2


st.write("Says President Barack Obama told a room of students, Children, every time I clap my hands together, a child in America dies from gun violence, and then a child told him he could solve the problem by not clapping any more.")
st.write("Topic: guns")
st.write("Speaker: chain-email")
option3 = st.selectbox(
    'True or False',
     ["I don't know", 'True', 'False'], key='option3')

'You think the statement is:', option3


st.write("Says Hillary Clinton wants to go to a single-payer plan for health care")
st.write("Topic: health-care")
st.write("Speaker: Donald Trump")
st.write('Job: President-Elect')
option4 = st.selectbox(
    'True or False',
     ["I don't know", 'True', 'False'], key='option4')

st.write("Ninety-seven percent of Americans do not receive subsidies for health care under the Affordable Care Act.")
st.write("Topic: congress,government-regulation,guns,public-health,states")
st.write("Speaker: Austin Scott")
option5 = st.selectbox(
    'True or False',
     ["I don't know", 'True', 'False'], key='option5')

st.write("There is no record of congresswoman Betty Sutton ... ever holding a single in-person town hall meeting open to the general public.")
st.write("Topic: job-accomplishments")
st.write("Speaker: Jim Renacci")
st.write("Job: U.S. representative")
st.write("Party: Republican")
option6 = st.selectbox(
    'True or False',
     ["I don't know", 'True', 'False'], key='option6')


fact_list=np.array(['False', 'True', 'False', 'False', 'True', 'False'])
check=st.button('check my answer', key='check')
pred_list=np.array([option1, option2, option3, option4, option5, option6])
if check:
    st.write(f'you have got {sum(pred_list==fact_list)}/6 correct')
    st.write('The answer is true, false, false, false, true, false')



st.header('Understanding Fake News and How Models Distinguish Them')

with open('preprocess/Introduction.txt', 'r') as f:
    intro_text=f.readlines()[0]

st.write(intro_text)
st.header('On what subjects do people tend to lie?')


columns=['id','label','statement','subject','speaker', 'job', 'state','party','barely_true_counts','false_counts',
                  'half_true_counts','mostly_true_counts','pants_on_fire_counts','context']
# label_values=['false','pants-fire','barely-true','true','mostly-true','false','half-true']
label_values=['false','true']
meta_feature=['subject','speaker', 'job', 'state','party']


def one_hot_encoding(x):
    if x=='true' or x=='mostly-true' or x=='half-true':
        return 'true'
    else:
        return 'false'
    
@st.cache(allow_output_mutation=True)
def read_data():
    
    df_train=pd.read_csv('liar_dataset/train.tsv', delimiter='\t', header=None, names=columns)
    df_test=pd.read_csv('liar_dataset/test.tsv', delimiter='\t', header=None, names=columns)
    df_valid=pd.read_csv('liar_dataset/valid.tsv', delimiter='\t', header=None, names=columns)
    df_total=pd.concat([df_train, df_test, df_valid]).reset_index(drop=True)
    df_train['label']=df_total.label.apply(lambda x: one_hot_encoding(x))
    df_test['label']=df_total.label.apply(lambda x: one_hot_encoding(x))
    df_valid['label']=df_total.label.apply(lambda x: one_hot_encoding(x))
    df_total['label']=df_total.label.apply(lambda x: one_hot_encoding(x))
    return df_train, df_test, df_valid, df_total

df_train, df_test, df_valid, df_total=read_data()

df_train.label.value_counts()

st.write('Example news statement from LIAR dataset')
df_train.statement[:10]

# combine_labels=st.checkbox('Combined labels')

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

@st.cache(allow_output_mutation=True)
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
combined_table.sort_values(by=['variable', 'kind'], inplace=True)
# if combine_labels:
scatter_chart=st.altair_chart(
    alt.Chart(combined_table, width=700).mark_bar().encode(
        x='value:Q',
        y=alt.Y('kind:N', sort='-x'),
        color='variable:N'
    ).interactive()
)

# else:
#     label_sel=st.selectbox(
#         'Select a label value: ',
#          label_values)

#     # combined_table=meta_feature_filtering(df_train, top_n, label_sel, feature_sel)
#     combined_table=combined_table.loc[combined_table.variable==label_sel]
#     combined_table=combined_table.drop('variable', axis=1)

#     scatter_chart=st.altair_chart(
#         alt.Chart(combined_table).mark_bar().encode(
#             x='value:Q',
#             y=alt.Y('kind:N', sort='-x')
#         ).interactive()
#     )

########################### PART 2 ##############################

st.header('Word Frequency for news with different topics')

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

@st.cache
def preprocess_statement(df):
    df['new_statement']=df.statement.apply(lambda x: text_lowercase(x))
    df['new_statement']=df.new_statement.apply(lambda x: remove_punctuation(x))
    df['new_statement']=df.new_statement.apply(lambda x: remove_whitespace(x))
    df['new_statement']=df.new_statement.apply(lambda x: remove_stopwords(x))
    df['new_statement']=df.new_statement.apply(lambda x: stem_words(x))
    df['new_statement']=df.new_statement.apply(lambda x: lemmatize_word(x))
    return df

@st.cache
def extract_key_words(df, label, meta_feature, value):
    sub_df=df.loc[df.label==label]
    sub_df=sub_df.loc[sub_df[meta_feature]==value].reset_index(drop=True)
    total_text=sub_df.new_statement.sum()
    return total_text


feature_sel=st.selectbox(
    'Select a Meta Feature: ',
     meta_feature, key='meta_feature2')

top_ten_subjects=df_train[feature_sel].value_counts()[:20].index

df_train=preprocess_statement(df_train)

label_sel=st.selectbox(
        'Select a label value: ',
         label_values, key='label_sel2')

subject_type=st.selectbox(
        'Select a value for the meta feature: ',
         top_ten_subjects)

total_text=extract_key_words(df_train, label_sel, feature_sel, subject_type)

try:
    fig=plt.figure(figsize=(10, 5))
    wordcloud=WordCloud(stopwords=stopwords, background_color="white").generate(total_text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    st.pyplot(fig)
except:
    st.write('No news found in the database, please try other selection criteria')


########################### PART 3 ##############################

st.header('Overall Sentiment analysis')


sid=SentimentIntensityAnalyzer()
@st.cache
def analyze_sentiment(sentence):
    score=sid.polarity_scores(sentence)
#     for k in sorted(score):
#         print('{0}: {1}, '.format(k, score[k]), end='')
#     print()
    return score

@st.cache
def sentiment_for_news(df, label, meta_feature, value):
    sub_df=df.loc[df.label==label]
    sub_df=sub_df.loc[sub_df[meta_feature]==value].reset_index(drop=True)
    scores={}
    for i in range(0, len(sub_df)):
        score=analyze_sentiment(sub_df.statement[i])
        scores['compound']=scores.get('compound',[])+[score['compound']]
        scores['negative']=scores.get('negative',[])+[score['neg']]
        scores['neutral']=scores.get('neutral',[])+[score['neu']]
        scores['positive']=scores.get('positive',[])+[score['pos']]
    return pd.DataFrame.from_dict(scores)

@st.cache
def sentiment_for_type(df, sentiment_type):
    scores={}
    for v in label_values:
        sub_df=df.loc[df.label==v].reset_index(drop=True)
        for i in range(0, len(sub_df)):
            score=analyze_sentiment(sub_df.statement[i])
            scores[v]=scores.get(v,[])+[score[sentiment_type]]
        
    return scores

sentiment_type=['compound','neg','neu','pos']
sentiment_type_table=sentiment_for_news(df_train, label_sel, feature_sel, subject_type)
# sentiment_type_table


scatter_chart=st.altair_chart(
    alt.Chart(sentiment_type_table).transform_fold(
        ['compound',
         'negative',
         'neutral',
         'positive'],
        as_ = ['Sentiments', 'value']
    ).transform_density(
        density='value',
        bandwidth=0.3,
        groupby=['Sentiments'],
        extent= [-2, 3]
    ).mark_area().encode(
        alt.X('value:Q'),
        alt.Y('density:Q', axis=None),
        alt.Row('Sentiments:N')
        
    ).properties(width=400, height=50).interactive()
)




########################### PART 4 ##############################
st.header('Other text statistics for news')

import readability
def get_other_statistics(df, label, if_filter, meta_feature, value):
    readabilty_grades_list=[]
    sentence_info_list=[]
    word_usage_list=[]
    sentence_beginnings_list=[]
    for l in label:
        readabilty_grades=[]
        sentence_info=[]
        word_usage=[]
        sentence_beginnings=[]
        
        sub_df=df.loc[df.label==l].reset_index(drop=True)
        if if_filter:
            sub_df=sub_df.loc[sub_df[meta_feature]==value].reset_index(drop=True)
            
        # loop through all labels
        for i in range(0, len(sub_df)):
            results=readability.getmeasures(df_train.new_statement[i], lang='en')
            readabilty_grades.append(results['readability grades'])
            sentence_info.append(results['sentence info'])
            word_usage.append(results['word usage'])
            sentence_beginnings.append(results['sentence beginnings'])
        readabilty_grades=pd.DataFrame.from_dict(readabilty_grades)
        readabilty_grades['label']=l
        
        sentence_info=pd.DataFrame.from_dict(sentence_info)
        sentence_info['label']=l
        
        word_usage=pd.DataFrame.from_dict(word_usage)
        word_usage['label']=l
        
        sentence_beginnings=pd.DataFrame.from_dict(sentence_beginnings)
        sentence_beginnings['label']=l
        
        readabilty_grades_list.append(readabilty_grades)
        sentence_info_list.append(sentence_info)
        word_usage_list.append(word_usage)
        sentence_beginnings_list.append(sentence_beginnings)
    
    readabilty_grades_list=pd.concat(readabilty_grades_list)
    sentence_info_list=pd.concat(sentence_info_list)
    word_usage_list=pd.concat(word_usage_list)
    sentence_beginnings_list=pd.concat(sentence_beginnings_list)
    
    return readabilty_grades_list, sentence_info_list, word_usage_list, sentence_beginnings_list

@st.cache
def select_df(var1, var2, subcategory, statistic_df, if_filter, n):
    df=statistic_df[subcategory]
    sub_df=df[[var1, var2, 'label']]
    if not if_filter:
        return sub_df.sample(n=n, random_state=1)
    return sub_df

filter=st.checkbox('Adding filtering criteria for the data, if no criteria, then n randomly selected data point will be displayed')

if not filter:
    n_point=st.slider('Select the number of points to display', 1, 5000)
else:
    n_point=500
    feature_sel=st.selectbox(
        'Select a Meta Feature: ',
         meta_feature, key='meta_feature3')

    top_ten_subjects=df_train[feature_sel].value_counts()[:20].index

    subject_type=st.selectbox(
            'Select a value for the meta feature: ',
             top_ten_subjects, key='value3')

readabilty_grades, sentence_info, word_usage, sentence_beginnings=get_other_statistics(df_train, label_values, filter,feature_sel, subject_type)

statistics_type=['sentence_info', 'readabilty_grades','word_usage','sentence_beginnings']
statistic_columns={'sentence_info':list(sentence_info.columns[:-1]), 'readabilty_grades': list(readabilty_grades.columns[:-1]), 
                  'word_usage': list(word_usage.columns[:-1]), 'sentence_beginnings': list(sentence_beginnings.columns[:-1])}
statistic_df={'sentence_info':sentence_info, 'readabilty_grades': readabilty_grades, 
                  'word_usage': word_usage, 'sentence_beginnings': sentence_beginnings}

stat_type=st.selectbox('Select a text statistics for exploring', statistics_type)

var1=st.selectbox('Select frist metric for exploring', statistic_columns[stat_type])
var2=st.selectbox('Select second metric for exploring', statistic_columns[stat_type])

current_df=select_df(var1,var2, stat_type, statistic_df, filter, n_point)

brush = alt.selection(type='interval')
points = alt.Chart(current_df).mark_point().encode(
    x='FleschReadingEase:Q',
    y='Kincaid:Q',
    color=alt.condition(brush, 'label:N', alt.value('lightgray'))
).add_selection(
    brush
)
bars = alt.Chart(current_df).mark_bar().encode(
    y='label:N',
    color='label:N',
    x='count(label):Q'
).transform_filter(
    brush
)

brush=alt.selection(type='interval')
points=alt.Chart(current_df).mark_point().encode(
    x=var1,
    y=var2,
    color=alt.condition(brush, 'label:N', alt.value('lightgray'))
).add_selection(
    brush
)
bars=alt.Chart(current_df).mark_bar().encode(
    y='label:N',
    color='label:N',
    x='count(label):Q'
).transform_filter(
    brush
)

scatter_chart=st.altair_chart(
    points&bars
)

########################### PART 5 ##############################
st.header('How machine learning models distinguish fake news?')
st.write('There are have been different online fake news detector aiming to help people distinguish fake news from true news. Understanding how models predict fake news would also help people improving their ability in detecting fake news.')
st.write('However, there are little studies on how machine learning models actually predict the fake news and what part of a news have a greater contribution to the model decision. ')
st.write('We aim to make use of interpretable machine learning techniques to help people understand what words or phrases in a sentence that may cause the news to be predicted as true or fake. ')

