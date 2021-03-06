import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
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
import random
import pickle
import spacy_streamlit
import spacy

import plotly.express as px
from sklearn.manifold import TSNE
from collections import Counter
nlp = spacy.load('en_core_web_sm')

st.set_page_config(
         page_title="Fake New Detection",
         page_icon="🗞️",
         initial_sidebar_state="collapsed",
     )


nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
stopwords=nltk.corpus.stopwords.words('english')

st.markdown(
    f"""
        <h1 style="font-family: Gill Sans; font-weight: 700; font-size: 48px;">Learn to Predict Fake News</h1>
        """,
    unsafe_allow_html=True,
)



st.markdown(
        f'''
    <h2 style="font-family: Gill Sans; font-weight: 200; font-size: 30px;">Let's start with a small test! </h2>
    ''',
        unsafe_allow_html=True,
    )

st.markdown(
        f'''
    <h2 style="font-family: Gill Sans; font-weight: 200; font-size: 20px;">Could you correctly predict whether the following news statements are true or fake? </h2>
    ''',
        unsafe_allow_html=True,
    )
st.markdown(

    f'''<div style="text-align: center;font-family: Gill Sans;border-radius: 25px;padding: 20px;width:680px;height:100px;border:3px solid #000;">Says 57 percent of federal spending goes to the military and just 1 percent goes to food and agriculture, including food stamps. federal-budget, military, poverty</div>''',
        unsafe_allow_html=True,
    )


def set_collapse(Topics,Speaker=None,Job=None,Party=None):
    with st.beta_expander("See More Details"):
     st.markdown(f'''
        <div style=""><span>Topics:</span> <span style="background-color: #B8B9B5;">economy</span>, <span style="background-color: #FDAAAA">financial-regulation</span></div>
         <div style=""><span>Speaker:</span> <span style="background-color: #AACEFD;">dennis-kucinich</span></div>
         <div style=""><span>Job:</span> <span style="background-color: #B7FDF8;">U.S. representative</span></div>
         <div style=""><span>Party:</span> <span style="background-color: #F7FDB7;">U.S. democrat</span></div>
     ''',
        unsafe_allow_html=True,
        )


with st.beta_expander("See More Details"):
     st.markdown(f"""
        <div style="font-family: Gill Sans;"><span>Topics:</span> <span style="background-color: #B8B9B5;">federal-budget</span>, <span style="background-color: #FDAAAA">military</span>, <span style="background-color: #FDD3AA">poverty</span></div>
         <div style="font-family: Gill Sans;"><span>Speaker:</span> <span style="background-color: #AACEFD;">facebook-posts</span></div>
         <div style="font-family: Gill Sans;"><span>Job:</span> <span style="background-color: #B7FDF8;">Social media posting</span></div>
        
     """,
        unsafe_allow_html=True,
        )


option1 = st.selectbox(
    'True or False',
     ["I don't know", 'True', 'False'], key="option1")
st.markdown(

    f'''<div style="text-align: center;font-family: Gill Sans;border-radius: 25px;padding: 20px;width:680px;height:100px;border:3px solid #000;">The Fed created $1.2 trillion out of nothing, gave it to banks, and some of them foreign banks, so that they could stabilize their operations.</div>''',
        unsafe_allow_html=True,
)

with st.beta_expander("See More Details"):
     st.markdown(f"""
        <div style="font-family: Gill Sans;"><span>Topics:</span> <span style="background-color: #F7FDB7;">economy</span>, <span style="background-color: #B8B9B5">financial-regulation</span></div>
         <div style="font-family: Gill Sans;"><span>Speaker:</span> <span style="background-color: #FDAAAA;">dennis-kucinich</span></div>
         <div style="font-family: Gill Sans;"><span>Job:</span> <span style="background-color: #B7FDF8;">U.S. representative</span></div>
         <div style="font-family: Gill Sans;"><span>Party:</span> <span style="background-color: #AACEFD;">U.S. democrat</span></div>
        
     """,
        unsafe_allow_html=True,
        )
option2 = st.selectbox(
    'True or False',
     ["I don't know", 'True', 'False'], key="option2")

# 'Your answer is:', option2
st.markdown(

    f'''<div style="text-align: center;font-family: Gill Sans;border-radius: 25px;padding: 10px;width:680px;height:100px;border:3px solid #000;">Says President Barack Obama told a room of students, Children, every time I clap my hands together, a child in America dies from gun violence, and then a child told him he could solve the problem by not clapping any more.</div>''',
        unsafe_allow_html=True,
)

with st.beta_expander("See More Details"):
     st.markdown(f"""
        <div style="font-family: Gill Sans;"><span>Topics:</span> <span style="background-color: #DFE3E4;">guns</span></div>
         <div style="font-family: Gill Sans;"><span>Speaker:</span> <span style="background-color: #FDB7F6;">chain-email</span></div>
        
     """,
        unsafe_allow_html=True,
        )
option3 = st.selectbox(
    'True or False',
     ["I don't know", 'True', 'False'], key='option3')


fact_list=np.array(['False', 'True', 'False'])# 'False', 'True', 'False'

col1, col2, col3 = st.beta_columns(3)
check=None  
def assign_color(select,gold):
    if select==gold:
        return "#6AFD7B"
    else:
        return '#FB9186'

with col1:
    check=st.button('check my answer', key='check')
    pred_list=np.array([option1, option2, option3])#, option4, option5, option6
with col2:
    if check:
        st.markdown(f'<div style="font-family: Gill Sans;">you got {sum(pred_list==fact_list)}/3 correct</div>',unsafe_allow_html=True,)
        # st.write('The answer is true, false, false')#, false, true, false
with col3:
    if check:
        st.markdown(f"""
        <div style="font-family: Gill Sans;"><span>The answer is:</span> <span style="background-color: {assign_color(option1,'False')};">False</span>,<span style="background-color:{assign_color(option2,'True')} ;">True</span>,<span style="background-color: {assign_color(option3,'False')};">false</span></div>
        
     """,
        unsafe_allow_html=True,
        )
        # st.write(' true, false, false')#, false, true, false

st.markdown(
        f'''
    <h2 style="font-family: Gill Sans; font-weight: 200; font-size: 30px;">Understanding Fake News</h2>
    ''',
        unsafe_allow_html=True,
    )

st.markdown(f"""
        <div style="font-family: Gill Sans;text-align: justify;">From 2016's presidential election to this year's election, the public has noticed that intentionally spreading misleading information is much more prevalent than before. Many articles from well known news sources, as well as academic papers, have mentioned such problems. However, detecting fake news, especially for politics, is much harder than we think. One reason is that since this news includes various topics, to identify such news, people need to be an expert on the subjects that are being discussed or be very familiar with domain knowledge.  Furthermore, since fake news is intended to misguide public opinions, the languages and contents are often well designed, and such properties are not limited to political fake news. The prevalence of fake news did way more harm than we thought. As mentioned in one of the paper from AAAS:
<li><i>the prevalence of "fake news" has increased political polarization, decreased trust in public institutions, and undermined democracy.</i></li>
Ability to identify fake news and understand their characteristics are critically needed. Taking the advantages of big data, here we will investigate the characteristics of political fake news from different perspectives using one of the fake news benchmark dataset: LIAR. We hope after reading this article, you could be more confident in identifying fake news and have a sharper sense in identifying fake political news.</div>
        
     """,
        unsafe_allow_html=True,
        )
        
st.markdown(
        f'''
    <h2 style="font-family: Gill Sans; font-weight: 200; font-size: 30px;">A Fake News Dataset</h2>
    ''',
        unsafe_allow_html=True,
    )


st.markdown(
        f'''
    <p style="text-align: justify;font-family: Gill Sans">In our article, we develop the analysis based on the LIAR dataset, a benchmark dataset for fake news detection. It is collected from POLITIFACT.COM and annotated using human labor, which guarantes the accuracy. The dataset contains over 10000 news with six classes, namely: true, mostly-true, half-true, barely-true, false, and pants-fire. These represent six degrees of authenticity. In our analysis, we binned those labels into two groups: True News (including labels: true, mostly-true, half-true) and False News (including labels: barely-true, false, and pants-fire). </p>
    ''',
        unsafe_allow_html=True,
    )

st.markdown(
        f'''
    <h2 style="font-family: Gill Sans; font-weight: 200; font-size: 30px;">What is fake news about and who tells fake news?</h2>
    ''',
        unsafe_allow_html=True,
    )

st.markdown(
        f'''
    <p style="text-align: justify;font-family: Gill Sans"> The proliferation of fake news in every day access social media platform, online news blogs and online newspaper mades it hard for individuals to check the reliability of the online contents. Fake news are deliberately constructed stories that intend to misguide public opinion and to seek financial gain. To understand what subjects fake news discussed and who tells fake news, we developed the interactive figure below to explore which are more often associated with fake news and who tell the most amount of fake news.   </p>
    ''',
        unsafe_allow_html=True,
    )
    
st.markdown(
        f'''
    <p style="text-align: justify;font-family: Gill Sans">  Through analysis, it is found that the subjects with most amount of fake news include health-care, taxes, immigration, elections and candidates-biography. Given the large amount of news that are associated with those topics, we also calculated the fake news ratio, number of fake news to total number of news, for further analysis and it is found that subjects with highest fake news ratio are state-budget, labor (over 80%), terrorism and foreign-policy (over 70%), and health-care and religion, over 60%.   </p>
    ''',
        unsafe_allow_html=True,
    )
st.markdown(
        f'''
    <p style="text-align: justify;font-family: Gill Sans">  The source of fake news is important factor in determining whether a news content is reliable or not. By looking at speakers that have the highest fake news ratio, we found that news from chain-email and blog posting are likely to be fake. Additionally, some speakers such as Donald Trump, Ben Carson, and Rush Limebaugh also have high ratio for telling fake news. </p>
    ''',
        unsafe_allow_html=True,
    )    

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
    # df_test=pd.read_csv('liar_dataset/test.tsv', delimiter='\t', header=None, names=columns)
    # df_valid=pd.read_csv('liar_dataset/valid.tsv', delimiter='\t', header=None, names=columns)
    # df_total=pd.concat([df_train, df_test, df_valid]).reset_index(drop=True)
    df_train['label']=df_train.label.apply(lambda x: one_hot_encoding(x))
    # df_test['label']=df_test.label.apply(lambda x: one_hot_encoding(x))
    # df_valid['label']=df_valid.label.apply(lambda x: one_hot_encoding(x))
    # df_total['label']=df_total.label.apply(lambda x: one_hot_encoding(x))
    # return df_train, df_test, df_valid, df_total
    return df_train

# df_train, df_test, df_valid, df_total=read_data()
df_train=read_data()

st.markdown(
        f'''
    <h2 style="font-family: Gill Sans; font-weight: 200; font-size: 20px;">Example news statement from LIAR dataset</h2>
    ''',
        unsafe_allow_html=True,
    )

# st.write('Example news statement from LIAR dataset')
df_train.statement[:10]

# combine_labels=st.checkbox('Combined labels')

top_n=st.slider(
    'Select the number of entries to show',
     1, 20,7)


kind=['subject','speaker', 'job', 'state']
def meta_feature_filtering(df, top_n, label, feature_sel):
    df_sub=df_train.loc[df_train.label==label]
    sel=pd.DataFrame(df_sub[feature_sel].value_counts()).reset_index()
    sel.columns=['kind','count']
    return sel

display_type=['Absolute', 'Percentage']
absolute=st.selectbox(
        'Select type of values: ',
        display_type)

feature_sel=st.selectbox(
    'Select a feature to filter the news: ',
     meta_feature)

@st.cache(allow_output_mutation=True)
def meta_feature_filtering_combined(df, top_n, kind, absolute):
    total=[]
    for l in label_values:
        df_sub=df_train.loc[df_train.label==l]
        sel=pd.DataFrame(df_sub[kind].value_counts()).reset_index()
        total.append(sel)
    combined_table=pd.DataFrame(total[0]).merge(pd.DataFrame(total[1]), on='index')
    combined_table.columns=[kind,label_values[0], label_values[1]]

    if absolute=='Percentage':
        combined_table['sum']=combined_table.sum(axis=1)
        combined_table=combined_table.loc[combined_table['sum']>20]
        combined_table.iloc[:, 1:-1]=combined_table.iloc[:, 1:-1].div(combined_table['sum'], axis=0)
        combined_table.sort_values(by=['false','sum'], inplace=True, ascending=False)
        combined_table=combined_table.drop('sum', axis=1)
        combined_table=combined_table[:top_n]
        combined_table=combined_table.melt(id_vars=kind)
    else:
        combined_table=combined_table[:top_n]
        combined_table=combined_table.melt(id_vars=kind)
    x1=combined_table.loc[combined_table.variable=='false']
    return combined_table, x1


combined_table_bar, x1=meta_feature_filtering_combined(df_train, top_n, feature_sel, absolute)

# if combine_labels:
scatter_chart=st.altair_chart(
    alt.Chart(combined_table_bar, width=700, title=f"Top {feature_sel} with highest number of fake news ({absolute})").mark_bar().encode(
         x='value:Q',
         y=alt.Y(f'{feature_sel}:N', sort=list(x1[feature_sel])),
    color='variable:N',
    ).interactive()
)

########################### PART 5 ##############################



@st.cache
def get_sentences():
    with open(r"add_csv/True_sentence.pickle", "rb") as input_file:
        True_sentence = pickle.load(input_file)
    with open(r"add_csv/False_sentence.pickle", "rb") as input_file:
        False_sentence = pickle.load(input_file)
    return True_sentence,False_sentence

st.markdown(
        f'''
    <h2 style="font-family: Gill Sans; font-weight: 200; font-size: 30px;">Whom and Where are mentioned in Fake News?</h2>
    <p style="font-family: Gill Sans"> Understanding what subjects and people appear in the fake news statement is just as important as understanding who are telling them. In this section, we will use Name Entity Recognition techniques to find what are the frequently mentioned entities in both true and fake news. </p> 
    ''',
        unsafe_allow_html=True,
    )

st.markdown(
        f'''
    <h2 style="font-family: Gill Sans; font-weight: 200; font-size: 20px;">You can randomly select any sample sentence from both true news and fake news to visualize the tagged entities within them. Feel free to explore!</h2>
    ''',
        unsafe_allow_html=True,
    )


True_sentence,False_sentence=get_sentences()
t_f=st.selectbox(
    '',
    ['True News','False News'])
change=st.button('change sentence', key='change')
i=2
if change:
    i=random.randint(0,100)



if t_f=='False News':
    sentence_l=True_sentence
else:
    sentence_l=False_sentence
s=sentence_l[i]
docx = nlp(s)
spacy_streamlit.visualize_ner(docx,labels=nlp.get_pipe('ner').labels,show_table=False,title='')


@st.cache
def get_sub_csv():
    geo_df=pd.read_csv('add_csv/GEO.csv').drop('Unnamed: 0',axis=1)
    per_df=pd.read_csv('add_csv/PER.csv').drop('Unnamed: 0',axis=1)
    return geo_df,per_df
st.markdown(
        f'''
    <h2 style="font-family: Gill Sans; font-weight: 200; font-size: 20px;">Most Frequent Locations Mentioned in Both True and False News</h2>
    ''',
        unsafe_allow_html=True,
    )

geo_df,per_df=get_sub_csv()
# print(len(geo_df))
col1,col2=st.beta_columns(2)
geo_bar=alt.Chart(geo_df.melt(id_vars='index')).mark_bar().encode(
        x=alt.X('variable:N', axis=alt.Axis(title=None,labels=False)),
        y='value:Q',
        color=alt.Color('variable', scale=alt.Scale(scheme='set1')),
        column=alt.Column('index',
        header=alt.Header(titleOrient='bottom', labelOrient='bottom')

    )).properties(
    width=26,
    height=100
).interactive()
geo_bar
st.markdown(

    f'''<div style="text-align:justify;font-family: Gill Sans;">From the plot above, we can see that the United States is more frequently mentioned in both two types of news, and they have similar proportions as well. Such findings are expected, as most of the political news from politifact.com is from the US, and due to the large sample size, both two types of news are equally distributed. Furthermore, we can see news containing New Jersey, Rhode Island, and Washington D.C. are mostly true news. We can see that these two areas, namely the New York region and D.C. region, are regarded as an economic center and political center. The news related to these areas is more likely facts-based since the misinformation might be very easy to check.  Additionally, this could also explain why China and Iran have more fake news than true news. The information about these two countries and other foreign countries might be hard to verify due to the distance and language barrier. </div>''',
        unsafe_allow_html=True,
    )
st.markdown(
        f'''
    <h2 style="text-align:justify;font-family: Gill Sans; font-weight: 200; font-size: 20px;">Most Frequent Names Mentioned in Both True and False News</h2>
    ''',
        unsafe_allow_html=True,
    )
per_bar=alt.Chart(per_df.melt(id_vars='index')).mark_bar().encode(
        x=alt.X('variable:N', axis=alt.Axis(title=None,labels=False)),
        y='value:Q',
        color=alt.Color('variable', scale=alt.Scale(scheme='set1')),
        column=alt.Column('index',
        header=alt.Header(titleOrient='bottom', labelOrient='bottom')

    )).properties(
    width=40,
    height=100
).interactive()
per_bar
st.markdown(

    f'''<div style="text-align:justify;font-family: Gill Sans;">Here we extracted the most frequent names mentioned in both true and fake news. From the bar plot, we can see that Barack Obama and Hilary Clinton appear in more fake news statements than true news statements. Such findings align with findings in many news reports and studies. In 2016's election, both Hilary and Obama have publicly criticized the spread of fake news and mentioned the severe consequences it could bring to society. Besides, in the Standford paper “Social Media and Fake News in the 2016 Election,”  Professor Gentzkow pointed out: <li><i>"Trump’s victory has been dogged by claims that false news stories – including false reports that Hillary Clinton sold weapons to ISIS and the pope had endorsed Trump – altered the outcome".</i></li> Such claims confirm our findings that Hilary is a common target of fake news during 2016 election.</div>''',
        unsafe_allow_html=True,
    )
########################### PART 2 ##############################

st.markdown(
    """ <h2 style="font-family: Gill Sans; font-weight: 200; font-size: 30px;">Word Frequency for news with different topics</h2>""",
    unsafe_allow_html=True
    )
    
st.markdown(

    f'''<p style="text-align:justify;font-family:Gill Sans;">
In this section, we will explore the frequent words and words group used in each type of news base on their categories. Here we first extracted the most frequent words for both fake and true news and then extracted the word vector through mapping it with the GloVe embedding. Them we applied t-SNE (t-distributed stochastic neighbor embedding) to reduce the dimensionality to 2 so that we can visualize and measure the similarity between words in a 2D space. </p>

    ''',
 unsafe_allow_html=True,
    )

st.markdown(
    f'''<p style="text-align:justify;font-family:Gill Sans;">Word frequency for different types of news allows us to identify what words are likely to appear in fake news. By looking at news with different subjects, speakers, jobs, parties, and state, we could learn about words that show up in the fake news. For instance, if we use the subject as a filter and select news about crimes, we see that the word “gun” is likely to appear in fake news. Similarly, if we select news that is spoken by Donald Trump, we see the words “Clinton", “war", “bill” and “Iraq” appear very often. Moreover, from the bubble plot, we can see that the frequent words used in the fake news are relatively denser than the true news. For example independent-party, healthcare-subject, presidential-candidate-job. Such findings are intuitive, where the true news covers a wide range of topics while the fake news only focusing on some of those that are hard to validate.</p>
    ''',
 unsafe_allow_html=True,
    )
# st.write("(Click on true or false label to remove it from the plot)")
st.markdown(

    f'''<div style="text-align:justify;font-family:Gill Sans;color:#fc031c;"><li>Click on true or false label in legend to remove it from the plot</li></div>

    ''',
 unsafe_allow_html=True,
    )

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

@st.cache
def load_glove():
    with open('add_csv/glove.pickle','rb')as f:
        dic=pickle.load(f)
    return dic

@st.cache
def get_plot_df(t1,t2,label1,label2,embeddings_dict):
    n=50
    a_c=Counter(t1.split(' '))
    a_c1=Counter(t2.split(' '))
    word=[x[0] for x  in a_c.most_common(n) if x not in stopwords]
#     count_a=[x[1] for x  in a_c.most_common(n) if x not in stopwords]
    
    word1=[x[0] for x  in a_c1.most_common(n) if x not in stopwords]
#     count_b=[x[1] for x  in a_c1.most_common(n) if x not in stopwords]
    
    
    word_vec1={}
    word_vec2={}
    count={}
    count1={}
    c1,c2=0,0
    for x in word:
        if x in embeddings_dict:
            c1+=1
            word_vec1[x]=embeddings_dict[x]
            count[x]=a_c[x]
    for x in word1:
        if x in embeddings_dict:
            c2+=1
            word_vec2[x]=embeddings_dict[x]
            count1[x]=a_c1[x]
    tsne = TSNE(n_components=2, random_state=0)
    words1 =  list(word_vec1.keys())
    words2=list(word_vec2.keys())
    cs=list(count.values())+list(count1.values())
    vectors = [word_vec1[word] for word in words1]+[word_vec2[word] for word in words2]
    Y = tsne.fit_transform(vectors[:1000])
    dd=pd.DataFrame(Y,columns=['X','Y'])#,'Z'
    print(len(Y))
    dd['word']=words1+words2
    dd['count']=cs
    
    dd['label']=[label1]*c1+[label2]*c2
    temp={}
    for i in range(len(dd)):
        if dd.iloc[i].word in temp:
            dd.iloc[i]=[temp[dd.iloc[i].word][0],temp[dd.iloc[i].word][1],dd.iloc[i].word,dd.iloc[i]['count'],dd.iloc[i].label]
        else:
            temp[dd.iloc[i].word]=(dd.iloc[i].X,dd.iloc[i].Y)
    return dd
# def get_plot_df(t,label,embeddings_dict):
#     n=50
#     a_c=Counter(t.split(' '))
#     word=[x[0] for x  in a_c.most_common(n) if x not in stopwords]
#     count=[x[1] for x  in a_c.most_common(n) if x not in stopwords]
#     word_vec={}
#     count={}
#     for x in word:
#         if x in embeddings_dict:
#             word_vec[x]=embeddings_dict[x]
#             count[x]=a_c[x]
#     tsne = TSNE(n_components=2, random_state=0)
#     words =  list(word_vec.keys())
#     cs=list(count.values())
#     vectors = [word_vec[word] for word in words]
#     Y = tsne.fit_transform(vectors[:1000])
#     dd=pd.DataFrame(Y,columns=['X','Y'])#,'Z'
#     dd['word']=words
#     dd['count']=cs
#     dd['label']=[label]*len(dd)
#     return dd

feature_sel=st.selectbox(
    'Select a feature to analyze news on: ',
     meta_feature, key='meta_feature2')

top_ten_subjects=df_train[feature_sel].value_counts()[:20].index

df_train=preprocess_statement(df_train)
col1,col2=st.beta_columns(2)

subject_type1=st.selectbox(
        'Select a value for the feature: ',
         top_ten_subjects)

    
# print('*******',subject_type1,subject_type2,label_sel2,label_sel1)

glove=load_glove()
total_text1=extract_key_words(df_train, 'true', feature_sel, subject_type1)
total_text2=extract_key_words(df_train, 'false', feature_sel, subject_type1)
# sent1_df=get_plot_df(total_text1,label_sel1,glove)
# sent2_df=get_plot_df(total_text2,label_sel2,glove)
dd=get_plot_df(total_text1,total_text2,'true','false',glove)
# with col1:
fig = px.scatter(dd, x="X", y="Y",
             size="count", 
                 text="word" ,color='label',log_x=True, size_max=60)
st.plotly_chart(fig,width=20, height=400)
# with col2:
# fig = px.scatter(sent2_df, x="X", y="Y",
#              size="count", 
#                  text="word" ,log_x=True, size_max=60)
# st.plotly_chart(fig,width=50, height=400)

# try:
#     fig=plt.figure(figsize=(10, 5))
#     wordcloud=WordCloud(stopwords=stopwords, background_color="white").generate(total_text)
#     # print(WordCloud(stopwords=stopwords).process_text(total_text))
#     plt.imshow(wordcloud, interpolation='bilinear')
#    plt.axis("off")
#     plt.show()
#     st.pyplot(fig)
# except:
#     st.write('No news found in the database, please try other selection criteria')


########################### PART 3 ##############################

st.markdown(
    """ <h2 style="font-family: Gill Sans; font-weight: 200; font-size: 30px;">Sentiment analysis for news with different topics</h2>""",
    unsafe_allow_html=True

    )
sid=SentimentIntensityAnalyzer()

st.markdown(
    """ <p style="font-family: Gill Sans; text-align:justify;">Sentiment analysis allows people to quantify and study the subjective information of a given news statement. </p>""",
    unsafe_allow_html=True

    )
st.markdown(
    """ <p style="font-family: Gill Sans; text-align:justify;">In this section, we applied VADER (Valence Aware Dictionary and sEntiment Reasoner), a lexicon and rule-based tool in analyzing the overall sentiment distribution for fake news and true news. </p>""",
    unsafe_allow_html=True

    )
st.markdown(
    """ <p style="font-family: Gill Sans; text-align:justify;">In addition to positive, negative, and neutral sentiments, the Compound score is a score that calculates the sum of all the lexicon ratings which have been normalized between -1 (most extreme negative) and +1 (most extreme positive).</p>""",
    unsafe_allow_html=True

    )
st.markdown(
    """ <p style="font-family: Gill Sans; text-align:justify;">We observe different distributions for sentiments when applying filters on news data. For instance, we observe that for the news with a topic on healthcare, fake news has a more positive distribution compared to true news. And for the topics on abortion, the false news is less neutral compared to the true news.</p>""",
    unsafe_allow_html=True

    )
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
label_sel_sent=st.selectbox(
            'Select a label value: ',
             label_values, key='label_sel_sent')

subject_type_sent=st.selectbox(
        'Select a feature to filter the news: ',
         top_ten_subjects,key='sent')
sentiment_type=['compound','neg','neu','pos']
sentiment_type_table=sentiment_for_news(df_train, label_sel_sent, feature_sel, subject_type_sent)
# sentiment_type_table


scatter_chart=st.altair_chart(
    alt.Chart(sentiment_type_table).transform_fold(
        ['negative',
         'neutral',
         'positive',
         'compound'],
         # colors=["#00AFBB","#00AFBB", "#E7B800", "#FC4E07"],
        as_ = ['Sentiments', 'value']
    ).transform_density(
        density='value',
        bandwidth=0.3,
        groupby=['Sentiments'],
        extent= [-2, 3]
    ).mark_area().encode(

        alt.X('value:Q'),
        alt.Y('density:Q', axis=None),
        alt.Row('Sentiments:N'),
        color="Sentiments:N",
        
    ).properties(width=600, height=30).interactive()
)



########################### PART 4 ##############################
st.markdown(
    """ <h2 style="font-family: Gill Sans; font-weight: 200; font-size: 30px;">Sentence Level Information for news</h2>""",
    unsafe_allow_html=True

    )
    
    
st.markdown(
    """ <p style="font-family: Gill Sans; text-align:justify">We further examined the sentence level information for news, including the number of words in a sentence, number of characters per words, total characters, total words, number of complex words, number of long words, and type-token ratio (number of unique words to number of total words). Overall, there seem to be no obvious differences between fake news and true news on those features in our dataset. However, when we apply a filter on the dataset, we found that for different speakers, for instance, sentence-level features will be different for fake news and true news. If we filter the dataset with speakers such as Donald Trump, we will see that the number of complex words in a statement will be larger for fake news than for true news.</p>""",
    unsafe_allow_html=True

    )


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

@st.cache
def get_statistic_df(df_train, label_values, filter, feature_sel, subject_type_sent):
    readabilty_grades, sentence_info, word_usage, sentence_beginnings=get_other_statistics(df_train, label_values, filter,feature_sel, subject_type_sent)
    statistics_type=['sentence_info', 'readabilty_grades','word_usage','sentence_beginnings']
    statistic_columns={'sentence_info':list(sentence_info.columns[:-1]), 'readabilty_grades': list(readabilty_grades.columns[:-1]), 
                  'word_usage': list(word_usage.columns[:-1]), 'sentence_beginnings': list(sentence_beginnings.columns[:-1])}
    statistic_df={'sentence_info':sentence_info, 'readabilty_grades': readabilty_grades, 
                  'word_usage': word_usage, 'sentence_beginnings': sentence_beginnings}
    return statistic_df

filter=st.checkbox('Add filtering ')

if not filter:
    n_point=st.slider('Select the number of points to display', 1, 5000, 100)
    subject_type=None
    feature_sel=None
    statistic_df=get_statistic_df(df_train, label_values, filter, feature_sel, subject_type)
else:
    n_point=500
    feature_sel=st.selectbox( 'Select a filtering on news: ', meta_feature, key='meta_feature3')
    top_ten_subjects=df_train[feature_sel].value_counts()[:20].index
    subject_type=st.selectbox('Select a value for the meta feature: ', top_ten_subjects, key='value3')
    statistic_df=get_statistic_df(df_train, label_values, filter, feature_sel, subject_type)
    
var_to_name={"characters_per_word": 'average characters per word', 'characters':'number of total characters', 
          'syll_per_word': 'average syllables per word', 'words_per_sentence': 'average words per sentence', 'type_token_ratio': 'type token ratio',
         'syllables': 'number of total syllables', 'words': 'number of total words', 'wordtypes': 'number of word types',
         'long_words': 'number of long words', 'complex_words': 'number of complex words'}
name_to_var={v: k for k, v in var_to_name.items()}

DEFAULT1='average characters per word'
DEFAULT2='average words per sentence'

def selectbox_with_default1(text, values, default=DEFAULT1, sidebar=False):
    func = st.sidebar.selectbox if sidebar else st.selectbox
    return func(text, np.insert(np.array(values, object), 0, default))
    
def selectbox_with_default2(text, values, default=DEFAULT2, sidebar=False):
    func = st.sidebar.selectbox if sidebar else st.selectbox
    return func(text, np.insert(np.array(values, object), 0, default))

var1=selectbox_with_default1('Select first sentence information', list(name_to_var.keys()))
var2=selectbox_with_default2('Select second sentence information', list(name_to_var.keys()))
var1=name_to_var[var1]
var2=name_to_var[var2]


current_df=select_df(var1,var2, 'sentence_info', statistic_df, filter, n_point)
current_df.shape
# Configure the options common to all layers

brush = alt.selection(type='interval')
base = alt.Chart(current_df).add_selection(brush)

# Configure the points
points = base.mark_point(color='red').encode(
    x=alt.X(f'{var1}:Q', title=''),
    y=alt.Y(f'{var2}:Q', title=''),
    color=alt.condition(brush, 'label', alt.value('grey'))
)

# Configure the ticks
tick_axis = alt.Axis(labels=False, domain=False, ticks=False)

x_ticks = base.mark_tick().encode(
    alt.X(f'{var1}:Q', title=var_to_name[var1], axis=tick_axis),
    alt.Y('label', title='', axis=tick_axis),
    color=alt.condition(brush, 'label', alt.value('lightgrey'))
)
y_ticks = base.mark_tick().encode(
    alt.X('label', title='', axis=tick_axis),
    alt.Y(f'{var2}', title=var_to_name[var2], axis=tick_axis),
    color=alt.condition(brush, 'label', alt.value('lightgrey'))
)
bars = alt.Chart(current_df).mark_bar().encode(
    x=alt.X('label:N',title=''),
    color='label:N',
    y=alt.Y('count(label):Q', title='Number of News Records')
).transform_filter(
    brush
) 
scatter_chart=st.altair_chart(
    y_ticks | (points & x_ticks) | bars 
)


########################### PART 6 ##############################
# st.header('How machine learning models distinguish fake news?')
# st.write('There are have been different online fake news detector aiming to help people distinguish fake news from true news. Understanding how models predict fake news would also help people improving their ability in detecting fake news.')
# st.write('However, there are little studies on how machine learning models actually predict the fake news and what part of a news have a greater contribution to the model decision. ')
# st.write('We aim to make use of interpretable machine learning techniques to help people understand what words or phrases in a sentence that may cause the news to be predicted as true or fake. ')


st.markdown(
    """ <h2 style="font-family: Gill Sans; font-weight: 200; font-size: 30px;">How machine learning models distinguish fake news?</h2>""",
    unsafe_allow_html=True

    )


st.markdown(
    """ <p style="font-family: Gill Sans; text-align:justify">There are different online fake news detectors aiming to help people distinguish fake news from true news. Understanding how models predict fake news would also help us improve our ability in detecting fake news. However, there are few studies on how machine learning models actually predict the fake news and what part of the news have a greater contribution to the model decision. We aim to make use of interpretable machine learning techniques to help people understand what words or phrases in a sentence may cause the news to be predicted as true or fake.</p>""",
    unsafe_allow_html=True

    )
    
st.markdown(
    """ <p style="font-family: Gill Sans; text-align:justify">We first trained a fake news classification model with BERT as a base model on the news statements from LIAR dataset. We then applied lime as an interpretation tool to analyze which words in the sentence play important roles for the final prediction.</p>""",
    unsafe_allow_html=True

    )
st.markdown(
    """ <p style="font-family: Gill Sans; text-align:justify">Here we presented three figures. The first one is the probability of a news being fake or true predicted by the fake news classification model, the second one visualizes the weights of top five most important words in a two-sided bar chart, and the last one is the words in the news statement highlighted in different colors, a darker color representing a greater importance. After seeing all the properties of fake and true news, you can choose a sentence from the dataset to see if your prediction is the same as the prediction by the model. 
</p>""",
    unsafe_allow_html=True

    )
    
    
################################ load from pickle of the feature importance ##############
from lime import lime_text


import streamlit.components.v1 as components

news_n = st.number_input('Select a news to view the model analysis',min_value=1, max_value=25, value=2)

HtmlFile = open(f"Model/html/text_{news_n}.html", 'r', encoding='utf-8')
source_code = HtmlFile.read() 
components.html(source_code, height=400)

st.markdown(
    """<p style="font-family: Gill Sans; text-align:justify">By interpreting the fake news classification model that we trained, we found that the model will put larger weights on the person’s name when news contains names that are frequently mentioned in fake news, such as Obama. However, those highlighted texts are not the only indicator since our model is based on BERT and it makes the judgment based on the neighboring contextual information regarding different words in a sentence as well. </p>""",
    unsafe_allow_html=True,

    )
st.markdown(
        f'''
    <h2 style="font-family: Gill Sans; font-weight: 200; font-size: 30px;">Summary</h2>
    ''',
        unsafe_allow_html=True,
    )
    
st.markdown(
    """<p style="font-family: Gill Sans; text-align:justify">Fake news contains misleading information and deliberately constructed stories that intend to misguide public opinion and to seek financial gain. With the current wide use of social media, fake news could spread quickly causing even more people to share the news unknowingly.
People are susceptible to false messages and could be easily misled by the information and it is important for us to have the ability to identify fake news.</p>""",
    unsafe_allow_html=True,

    )
    
st.markdown(
    """ <p style="font-family: Gill Sans; text-align:justify">In this article, we have compared fake news with true news from a few perspectives, including the topics that fake news covered, the people and subjects that are usually mentioned in fake news statements, and the frequently used words in different types of fake statements. Additionally, sentiment analysis and sentence level information are also provided for comparing fake news and true news in different news statements. Furthermore, we also find several interesting findings. The first one is the source of the news, namely the speaker, the job of the speaker are important to identify fake news. The intent is also a very important factor, for example, the election-related or geopolitical-related fake news is higher than other, this news is intended to capture people's attention and change people's opinion. Thirdly, as opposed to fake news, the true news covers more topics within the same subject, whereas fake news is more target-oriented. Lastly, we also find that fake news used words more exaggerated than true news, such word 'billion' use a lot in fake news, whereas in true news 'million' is more used.</p>""",
    unsafe_allow_html=True,

    )
st.markdown(
    """ <p style="font-family: Gill Sans; text-align:justify">Additionally, as many existing fake news detectors have achieved satisfactory results on fake news detection, we choose the BERT to train and used as our prediction model. We also applied the LIME framework, Local interpretable model-agnostic explanations, provided by the package lime to learn how the machine learning model distinguish between fake news and true news. We visualized words in the sentence that contributes to the model decision with the hope that this will also help us improve our ability in spotting and identifying fake news.</p>""",
 unsafe_allow_html=True,

    )
# st.markdown(
#     """ <p style="font-family: Gill Sans; text-align:justify">The language of fake news is just as important as the source of the news. In real life, news coming from authority may be trustworthy, however, it is also possible that people lie about things in order to affect people's attributes and beliefs. </p>""",
#     unsafe_allow_html=True,
#    )
                    
st.markdown("""<div style="font-family: Gill Sans;text-align: justify;"> We further provide links to some useful fake news detector and facts checking website. </div> """,
         unsafe_allow_html=True)
                    
                    
st.markdown(
        f'''
    <h2 style="font-family: Gill Sans; font-weight: 200; font-size: 30px;">Some Useful Facts Checking Website</h2>
    ''',
        unsafe_allow_html=True,
    )


st.markdown("""<p style="font-family: Gill Sans; text-align:justify"><li><a href="https://www.politifact.com">PolitiFact</a>: PolitiFact is a fact-checking website that rates the accuracy of claims by elected officials and others on its Truth-O-Meter.</li></p>

                """, unsafe_allow_html=True)
st.markdown("""<p style="font-family: Gill Sans; text-align:justify"><li><a href="https://www.factcheck.org">FactCheck</a>: monitors the factual accuracy of what is said by major U.S. political players in the form of TV ads, debates, speeches, interviews and news releases.</li></p>""", unsafe_allow_html=True)
st.markdown("""<p style="font-family: Gill Sans; text-align:justify"><li><a href= https://www.snopes.com>Snopes</a>: The definitive Internet reference source for researching urban legends, folklore, myths, rumors, and misinformation.</li></p>""", unsafe_allow_html=True)


st.markdown(
        f'''
    <h2 style="font-family: Gill Sans; font-weight: 200; font-size: 30px;">Reference</h2>
    ''',
        unsafe_allow_html=True,
    )



st.markdown("""<p style="font-family: Gill Sans; text-align:justify"><li> University of Central Florida News | UCF Today. 2020. How Fake News Affects U.S. Elections </li></p>""",
        unsafe_allow_html=True,
    )


st.markdown("""<p style="font-family: Gill Sans; text-align:justify"><li> Stellino, M., 2020. 8 Resources To Detect Fake News. [online] Newscollab.org </li></p>""",
        unsafe_allow_html=True,
    )


st.markdown("""<p style="font-family: Gill Sans; text-align:justify"><li> Researchguides.library.wisc.edu. 2020. [online] </li></p>""",
        unsafe_allow_html=True,
    )


st.markdown("""<p style="font-family: Gill Sans; text-align:justify"><li> Allcott, H. and Gentzkow, M., 2017. Social media and fake news in the 2016 election. Journal of economic perspectives, 31(2), pp.211-36. </li></p>""",
        unsafe_allow_html=True,
    )


st.markdown("""<p style="font-family: Gill Sans; text-align:justify"><li> Turner, P.A., 2018. Respecting the Smears: Anti-Obama Folklore Anticipates Fake News. Journal of American Folklore, 131(522), pp.421-425. </li></p>""",
        unsafe_allow_html=True,
    )
                    
st.markdown("""<p style="font-family: Gill Sans; text-align:justify"><li> Cunha, E., Magno, G., Caetano, J., Teixeira, D. and Almeida, V., 2018, September. Fake news as we feel it: perception and conceptualization of the term “fake news” in the media. In International Conference on Social Informatics (pp. 151-166). Springer, Cham. </li></p>""",
        unsafe_allow_html=True,
    )
                    
