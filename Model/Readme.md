The ipynb contains code for training BERT base model for fake news classification (binary), and steps for using lime for visualizing the text importance. 

We developed two different fake news prediction model, one with meta data and justification, achieving accuracies of over 70% on the LIAR dataset for both training and testing data. Another model only takes in the text statement (without meta data and justification), and we used the model for explaining the word importance for interpreting how the model makes a decision. 

The trained BERT model is too large for uploading to github, therefore we use the model as a predictor to predict on randomly selected news and extract word importance using lime package. 
Additionally, a link to google drive the contains the trained model is included: https://drive.google.com/file/d/1Tfun3DEPHyx83QfeuHzzl5haC_xkFSJI/view?usp=sharing 

The html folder contains the lime visualization of fake news classification model prediction results. 
Each html file contains a visualization of a news statement. 

