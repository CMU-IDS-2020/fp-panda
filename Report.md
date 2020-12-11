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

### Visualization Components

Our narrative article contains different interactive visualizations and those visualizations are intended to guide people to explore possible characteristics of fake news compared to true news. The article starts with three news statements, with some additional information, ie. topics, speaker, the job of speaker, party,  intending to test people’s ability to identify fake news. Through the test, we want to demonstrate that fake news could be very well-designed and people may have difficulties distinguishing fake news from true news if they are not an expert in the fields that are being discussed.

Then, we explored what topics fake news tends to cover and what people or media outlets often tell fake news through the analysis of the number of fake news and the ratio of fake news of different topics and speakers. For interaction, we allow users to select the number of entries to display and a feature for analyzing the news. 

Understanding what subjects and people appear in the fake news statement is just as important as understanding who is telling them. So in the next visualization, we applied Name Entity Recognition techniques to label the frequent entities mentioned in both fake news and true news. Readers could randomly choose a news statement from true news statements or fake news statements to read about the news with annotated entities. Afterward, we summarized the most frequently appeared places and names in both fake news and true news.  This allows readers to learn more about what names and places are likely to appear in a fake news story. 

Understanding what are the frequently used words in fake news and true news will also help people identify unreal statements. So we then explore the frequent words and word groups used in different types of the news based on their subjects, speakers, etc. We first extracted the most frequent words for both fake and true news and then extracted the word vector by mapping it with the GloVe embedding. Then we applied t-SNE (t-distributed stochastic neighbor embedding) to reduce the dimensionality to 2 so that we can visualize and measure the similarity between words in a 2D space. Similar to previous interactions, we also allow users to select a feature they want to analyze the news on and extract the corresponding most frequent words. 

Moreover, sentiment analysis allows people to quantify and study the subjective information of a given news statement. So we applied VADER (Valence Aware Dictionary and sEntiment Reasoner), a lexicon and rule-based tool in analyzing the overall sentiment distribution for fake news and true news. There are three commonly known sentiments, positive, negative, and neutral. In our visualization, in addition to positive, negative, and neutral sentiments, the Compound score is also presented, which is a score that calculates the sum of all the lexicon ratings which have been normalized between -1 (most extreme negative) and +1 (most extreme positive). News covering different subjects, and news from different speakers may have different sentiments for both fake type and true type. For instance, a speaker may present more negative sentiments when telling fake news and fake news for a specific topic may also tend to be more positive or negative in sentiment. This allows readers to select and explore the sentiments for the news types that they are interested in and learn more about their characteristics. 

We further examined the sentence level information for news, including the number of words in a sentence, number of characters per words, total characters, total words, number of complex words, number of long words, and type-token ratio (number of unique words to number of total words). Similarly, people could select a feature to analyze the news on and review different features and their relations. The visualization intends to offer additional sentence-level information for understanding the characteristics of fake news.

### Fake News Detection Model

Different online fake news detectors are aiming to help people distinguish fake news from true news. Understanding how models predict fake news would also help us improve our ability in detecting fake news. However, there are few studies on how machine learning models predict the fake news and what part of a news statement will affect the model decision. We aim to make use of interpretable machine learning techniques to help people understand what words or phrases in a sentence may cause the news to be predicted as true or fake. 

We first trained a fake news classification model with BERT as a base model on the news statements from LIAR dataset. We then applied lime as an interpretation tool to analyze which words in the sentence play important roles for the final prediction. Since the saved model is too large to use in the designed application, we have included 25 example news statements from LIAR dataset, each with weights for words generated from the trained fake news detection model and lime-text interpretability tool. We allow readers to randomly select a news statement and visualize them in the article. 

There are three different parts of the visualization. The first one is the probability of a news being fake or true predicted by the fake news classification model, the second one visualizes the weights of top five most important words in a two-sided bar chart, and the last one is the words in the news statement highlighted in different colors, a darker color representing a larger importance. This visualization allow users to compare their judgements with the model’s judgement and investigate what words are important for judgements. By understanding how a machine learning model makes predictions based on a news statement, readers could also think about how they will interpret a news statement. 

## Results
Fake news contains misleading information and deliberately constructed stories that intend to misguide public opinion and to seek financial gain. With the current wide use of social media, fake news could spread quickly causing even more people to share the news unknowingly. People are susceptible to false messages and could be easily misled by the information and it is important for us to have the ability to identify fake news. 

In this article, we have compared fake news with true news from a few perspectives, including the common topics that fake news covered, the people and subjects mentioned in fake news statements, and the frequently used words in different types of fake statements. Additionally, sentiment analysis and sentence level information are also provided for comparing fake news and true news in different news statements. 
From our investigations on what topics and speakers are more often associated with fake news, it is found that the subjects with the most amount of fake news include health-care, taxes, immigration, elections, and candidates-biography. Given a large amount of news that is associated with those topics, we also calculated the fake news ratio, number of fake news to the total number of news, for further analysis, and it is found that subjects with the highest fake news ratio are state-budget, labor (over 80%), terrorism and foreign-policy (over 70%), and health-care and religion, over 60%. The source of fake news is an important factor in determining whether news content is reliable or not. By looking at speakers that have the highest fake news ratio, we found that news from chain-email and blog posting is likely to be fake. Additionally, some speakers such as Donald Trump, Ben Carson, and Rush Limebaugh also have a high ratio for telling fake news.

Through the analysis of places that are frequently mentioned in news, it is observed that the United States is more frequently mentioned in both two types of news, and they have similar proportions as well. Such findings are expected, as most of the political news from politifact.com is from the US, and due to the large sample size, both two types of news are equally distributed. Furthermore, we can see news containing New Jersey, Rhode Island, and Washington D.C. are mostly true news. We can see that these two areas, namely the New York region and D.C. region, are regarded as an economic center and political center. The news related to these areas is more likely facts-based since the misinformation might be very easy to check. Additionally, this could also explain why China and Iran have more fake news than true news. The information about these two countries and other foreign countries might be hard to verify due to the distance and language barrier.

Through the analysis of names that are frequent mentioned in news, we observe that Barack Obama and Hilary Clinton appear in more fake news statements than true news statements. Such findings align with findings in many news reports and studies. In 2016's election, both Hilary and Obama have publicly criticized the spread of fake news and mentioned the severe consequences it could bring to society. Besides, in the Standford paper “Social Media and Fake News in the 2016 Election,” Professor Gentzkow pointed out: "Trump’s victory has been dogged by claims that false news stories – including false reports that Hillary Clinton sold weapons to ISIS and the pope had endorsed Trump – altered the outcome". Such claims confirm our findings that Hilary is a common target of fake news, especially during election time of 2016.

Word frequency for different types of news allows us to identify what words are likely to appear in fake news. For instance, if we look at news about crimes, we see that the word “gun” is likely to appear in fake news. Similarly, if we select news that is spoken by Donald Trump, we see the words “Clinton", “war", “bill” and “Iraq” appear very often. Word usage for fake news and true news could have great differences, for instance, for the candidate-biography fake news, words such as “ever”, “pay”, “even”, “governor” only appear in fake news and not in true news. When reading news, the use of words and phrases could be an important factor in determining its trustworthiness. 

Through additionally sentiment analysis and sentence level information of the news, it is found that fake news from some specific topics may have different sentiments compared to true news. For instance, we observe that for the news with a topic on healthcare, fake news has a more positive distribution compared to true news. And for the topics on abortion, the false news is less neutral compared to the true news. Moreover, we found that for different speakers, sentence-level features will be different for fake news and true news. If we analyze the dataset with speakers such as Donald Trump, we will see that the number of complex words in a statement will be larger for fake news than for true news. Those subjects and speaker-specific findings are also informative as they are the commonly appeared topics in daily news and commonly present public speakers in real life. 

By interpreting the fake news classification model that we trained, we found that the model will put larger weights on a person’s name when news contains names that are frequently mentioned in fake news, such as Obama. However, those highlighted texts are not the only indicator since our model is based on BERT and it makes the judgment based on the neighboring contextual information regarding different words in a sentence as well.

Additionally, as many existing fake news detectors have achieved satisfactory results on fake news detection, we choose the BERT to train and use as our prediction model. We also applied the LIME framework, Local interpretable model-agnostic explanations, provided by the package lime to learn how the machine learning model distinguishes between fake news and true news. We visualized words in the sentence that contribute to the model decision with the hope that this will also help us improve our ability in spotting and identifying fake news.

## Discussion
Overall, we have identified a few interesting findings. The source of the news, namely the speaker, and the job of the speaker are important to identify fake news. The intent is also a very important factor, for example, the proportion of election-related or geopolitical-related fake news is higher than other, since those types of news are intended to capture people's attention and change people's opinion. Moreover, as opposed to fake news, the true news covers more topics within the same subject, whereas fake news is more target-oriented. The sentiments and sentence-level information also differs for fake news and true news and so is the word usage. Lastly, we also find that fake news uses words more exaggerated than true news, such as the word 'billion' is used a lot in fake news, whereas in true news 'million' is more used.

Additionally, as many existing fake news detectors have achieved satisfactory results on fake news detection, we choose the BERT to train and use as our prediction model. We also applied the LIME framework, Local interpretable model-agnostic explanations, provided by the package lime to learn how the machine learning model distinguishes between fake news and true news. We visualized words in the sentence that contribute to the model decision with the hope that this will also help us improve our ability in spotting and identifying fake news.

## Future Work

With the widespread of fake news in online social media outlets, fake news also exists in different forms and the source of news, speaker of the news are all important factors in determining the reliability of a news statement. In the article, we have analyzed different aspects of fake news,  from news topics and speakers to text-related features, the frequent term used, sentiments, entities in fake news, and developed fake news prediction models with interpretability tools to analyze how models make predictions. However, our analysis mainly focuses on political news and does not focus on the news with images and videos, for future work, a multimodal approach could be used to analyze other forms of news as well and not just text statements. 

## Reference

University of Central Florida News | UCF Today. 2020. How Fake News Affects U.S. Elections 

Stellino, M., 2020. 8 Resources To Detect Fake News. [online] Newscollab.org

Researchguides.library.wisc.edu. 2020. [online]

Allcott, H. and Gentzkow, M., 2017. Social media and fake news in the 2016 election. Journal of economic perspectives, 31(2), pp.211-36.

Turner, P.A., 2018. Respecting the Smears: Anti-Obama Folklore Anticipates Fake News. Journal of American Folklore, 131(522), pp.421-425.

Cunha, E., Magno, G., Caetano, J., Teixeira, D. and Almeida, V., 2018, September. Fake news as we feel it: perception and conceptualization of the term “fake news” in the media. In International Conference on Social Informatics (pp. 151-166). Springer, Cham.
