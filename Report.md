# Final Project Report


Final Project Report

Fake News Analysis and Prediction

Team members:
Faner Lin (fanerl@andrew.cmu.edu)

Shuaidong Pan (shuaidop@andrew.cmu.edu)




**Project URL**: https://share.streamlit.io/cmu-ids-2020/fp-panda/main/app.py

Short (~250 words) abstract of the concrete data science problem and how the solutions addresses the problem.

## Introduction

## Related Work
Mand journal articles have pointed out that fake news will affect elections and the spread of fake news will also affect people’s attitudes, beliefs, and even actual behaviors. Additionally, it is found that the spread of fake news on hot topics, such as race relations, gun rights, and immigration issues, during different events would also divide the nation and causing chaos. 
## Methods
Our methods could be mainly divided into two different parts: first, we designed different interactive components to study the characteristics of the fake news in the LIAR  dataset, second, we trained fake news detection models with BERT as a base model and applied interpretability tools to analyze words in a news statement that affect model’s prediction. More details will be discussed in the following section. 

##### Visualization Components

Our narrative article contains different interactive visualizations and those visualizations are intended to guide people to explore possible characteristics of fake news compared to true news. The article starts with three news statements, with some additional information, ie. topics, speaker, the job of speaker, party,  intending to test people’s ability to identify fake news. Through the test, we want to demonstrate that fake news could be very well-designed and people may have difficulties distinguishing fake news from true news if they are not an expert in the fields that are being discussed.

Then, we explored what topics fake news tends to cover and what people or media outlets often tell fake news through the analysis of the number of fake news and the ratio of fake news of different topics and speakers. For interaction, we allow users to select the number of entries to display and a feature for analyzing the news. 

Understanding what subjects and people appear in the fake news statement is just as important as understanding who is telling them. So in the next visualization, we applied Name Entity Recognition techniques to label the frequent entities mentioned in both fake news and true news. Readers could randomly choose a news statement from true news statements or fake news statements to read about the news with annotated entities. Afterward, we summarized the most frequently appeared places and names in both fake news and true news.  This allows readers to learn more about what names and places are likely to appear in a fake news story. 

Understanding what are the frequently used words in fake news and true news will also help people identify unreal statements. So we then explore the frequent words and word groups used in different types of the news based on their subjects, speakers, etc. We first extracted the most frequent words for both fake and true news and then extracted the word vector by mapping it with the GloVe embedding. Then we applied t-SNE (t-distributed stochastic neighbor embedding) to reduce the dimensionality to 2 so that we can visualize and measure the similarity between words in a 2D space. Similar to previous interactions, we also allow users to select a feature they want to analyze the news on and extract the corresponding most frequent words. 

Moreover, sentiment analysis allows people to quantify and study the subjective information of a given news statement. So we applied VADER (Valence Aware Dictionary and sEntiment Reasoner), a lexicon and rule-based tool in analyzing the overall sentiment distribution for fake news and true news. There are three commonly known sentiments, positive, negative, and neutral. In our visualization, in addition to positive, negative, and neutral sentiments, the Compound score is also presented, which is a score that calculates the sum of all the lexicon ratings which have been normalized between -1 (most extreme negative) and +1 (most extreme positive). News covering different subjects, and news from different speakers may have different sentiments for both fake type and true type. For instance, a speaker may present more negative sentiments when telling fake news and fake news for a specific topic may also tend to be more positive or negative in sentiment. This allows readers to select and explore the sentiments for the news types that they are interested in and learn more about their characteristics. 

We further examined the sentence level information for news, including the number of words in a sentence, number of characters per words, total characters, total words, number of complex words, number of long words, and type-token ratio (number of unique words to number of total words). Similarly, people could select a feature to analyze the news on and review different features and their relations. The visualization intends to offer additional sentence-level information for understanding the characteristics of fake news. 
## Results

## Discussion

## Future Work
