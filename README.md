# CMU Interactive Data Science Final Project

* **Online URL**: https://share.streamlit.io/cmu-ids-2020/fp-panda/main/app.py
* **Team members**:
  * Contact person: Faner Lin (fanerl@andrew.cmu.edu)
  * Shuaidong Pan (shuaidop@andrew.cmu.edu)
* **Track**: Narrative

* **Video**: [Video Link](https://drive.google.com/file/d/12OOjPr4vsez3BYTGE7dJyItBCHoGRN76/view?usp=sharing)

## Work distribution
Faner Lin: Faner researched related topics of fake news detection and discussed with Shuaidong to design different components for the narrative article. Additionally, Faner focused on performing exploratory data analysis on the fake news dataset and designed the component for exploring the number of fake news associated with different subjects, speakers, and generating interactive components for sentiment analysis for different news and sentence-level information for different news. Faner also trained fake news classification model with BERT as the base model and applied interpretability tools lime-text for analyzing the word importance for fake news classification and visualized the importance through highlighted text images. Finally, Faner also participated in composing the final report and constructing the narratives in the article.

Shuaidong Pan: Shuaidong researched related topics and discussed with Faner to design different components for the narrative article. Additionally, Shuaidong also focused on selecting fake news dataset and designing interactive components for entity recognition in news statements and word frequency. He used GloVe embedding and t-SNE to visualize the word frequency in 2d space and applied Name Entity Recognition techniques to extract names and places entities in news statements. Shuaidong also worked on designing and improving the layout of the article and researching related works. Finally, Shuaidong also participated in composing the final report and constructing the narratives in the article.

## Abstract
With the prevalence of fake news in online social media and online news, it is becoming more and more challenging for people to identify trustworthy news sources and determine the reliability of news content. In our article, we investigated the characteristics of political fake news by developing analysis and visualization based on the LIAR dataset, which contains different political news and information regarding the source of news. We found that the source of news and the entities mentioned in news are important factors in identifying the truthfulness of a given news statement. Additionally, we found that the word usage, sentiment, and other sentence-level information will be different for fake news and true news under different topics or speakers. We also developed a fake news classification model with BERT as a base model to predict labels for given news. By using the LIME framework, Local interpretable model-agnostic explanations, we visualize the words in a news statement that affect the model's decision. By learning the characteristics of fake news and understanding how the machine learning model makes decisions, readers will have more confidence in detecting and analyzing fake news.

## Summary Image

![GitHub Logo](/Image/p3.png)
![GitHub Logo](/Image/p4.png)

## Deliverables

### Proposal

- [x] The URL at the top of this readme needs to point to your application online. It should also list the names of the team members.
- [x] A completed proposal. The contact should submit it as a PDF on Canvas.

### Design review

- [x] Develop a prototype of your project.
- [x] Create a 5 minute video to demonstrate your project and lists any question you have for the course staff. The contact should submit the video on Canvas.

slides for design review: https://docs.google.com/presentation/d/1VTH1WSd9Kigl7f7B-tc7kV_f1Nr3A_MZOovPSc6XQ6U/edit?usp=sharing

### Final deliverables

- [x] All code for the project should be in the repo.
- [x] A 5 minute video demonstration.
- [x] Update Readme according to Canvas instructions.
- [x] A detailed project report. The contact should submit the video and report as a PDF on Canvas.

### Running Instructions
1. clone the github repo to local directory
2. download all packages and dependencies needed. 
<code> pip install -r requirements.txt </code>
3. Change path to the project directory, run 
<code> streamlit run app.py </code>

