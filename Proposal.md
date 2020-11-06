# Final Project Proposal

## Introduction and Problem Statement
With the current massive use of social media, information quality becomes an important issue when it comes to news reporting and other sources of factuality checking. Given the hot topics in fake news classification and toxic comments classification, many datasets have been collected, and they cover a wide range of domains. For example, the LIAR dataset was collected from Politifact and covers the political topics, while FEVER which covers scientific topics is collected from Wikipedia. Our project aims to investigate the properties of fake news across different social media platforms using the existing benchmark datasets, and the potential social media platforms include Twitter, Facebook, Wikipedia, and Reddit. 

## Proposed Solution Overview
We intend to provide a comprehensive analysis of different types of fake news on social media platforms and propose a model for fake news detection. The following steps will be taken:
* Conducting a comparative study between true news and fake news, including identifying the language features for different types of fake news, such as word frequencies and topics, through LDA, TF-IDF, Sentiment Vader Score, Readability Scores, and other metadata constructions with effective visualization
* Constructing and visualizing models for fake news detection, and identifying important features in text and images that affect the model’s decision

## Scope of the project
We aimed to make use of the Fakeddit multimodal dataset that contains over 1 million samples collected from Twitter and other datasets collected from Facebook and Reddit. In our study, we plan to identify similarities and differences of properties (such as text features, semantic features, sentiment features) of fake news across these platforms, and we also intend to analyze the differences of types of fake news for different domains, such as politics, scientific post, etc. 

In addition to the above-mentioned analysis, we intend to propose a model for text-based, image-based, and text-image-based fake news detection since in real-life, fake news sometimes appears as a text and image combination. In our projects, we aim to analyze the performance of multimodal models and their learning ability on the fake news classification task. Moreover, since such a task requires two model pipelines: language model (usually used models are BERT, LSTM, Attention) and image models (usually used are Restnet, EffNet), we could investigate the learning ability of different language-image model pairs using various fusion techniques. Ideally, we could have a clearer understanding of how such multimodal problems handle the cognition of the fake news classification task and our model will also allow users to conduct factuality checking. 

## Potential Datasets 
* Fakeddit https://github.com/entitize/Fakeddit
* LIAR https://github.com/thiagorainmaker77/liar_dataset
* BuzzFeedNews https://github.com/BuzzFeedNews
* FakeNewsNet https://github.com/KaiDMML/FakeNewsNet
