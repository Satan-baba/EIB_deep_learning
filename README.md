# EIB_deep_learning
Our work can be summarized in three main stages:
Training and optimizing the TextCNN model 
Augmenting and updating our data with new annual reports and the most recent Bloomberg scores
Using BERT classifier to classify a company's annual report into score buckets.
Using XLNet classifier to classify a company’s annual report into score buckets. 


TextCNN model:
We received a TextCNN model from the last group. They had used a skipgram model for embeddings and fed it to the CNN for score estimation with an MSE of 157
We changed the parameters of the CNN model and retrained the skipgram model with the following data preprocessing and cleaning:
Removing stopwords from the already stemmed text and retraining the skipgram model
Manually changing the parameters of the TextCNN model to check the performance
Our experiments with the TextCNN model brought the MSE down to 119. 

       3) Used pre-trained Glove embeddings to be fed to the CNN model, got an MSE of 120
4) Trained a GloVe model on annual reports and got an MSE of 125
	Our findings from this was:
Removing stopwords in the skipgram model improved the model’s accuracy
The TextCNN performs better with higher epochs when run with a GPU such as the NVIDIA TESLA 200 offered on google cloud platform

Data Augmentation and update:
Scraped new annual reports and bloomberg scores for 2019 (left out company presentation since some are in other languages and do not contain enough information, may need sustainability reports in the future) 
2) Added more climate-related keywords 
3) Added industries/number of keywords/length of texts to assess quality of the corpus



BERT Classifier:
We Categorized the bloomberg scores into categorical ranges

Used a pre-trained Bert transformer to tokenize and encode the text data
Used the bert-tensorflow library to build a classifier model with a final loss of 0.47

XLNet Classifier:
We categorized the bloomberg scores into five categorical ranges, the label is made in sequence from 0 to 4.

The number of examples having each label is shown below.

Used “XLNet-base-cased” pretrained model to tokenize the text data.
Used “Pad sequences” function in Keras to encode the tokens. 
Used “XLNetSequenceClassificationModel” pretrained model to classify the features. The f1-score is in the range of 0.21 to 0.25.  


Performance of the models and methodology
TextCNN (kernel_sizes=[3,4,5],filter_num=256) + skipgram (200 dimensions) to predict new scores with data preprocessing gives an MSE of 119 
BERT gives a final loss of 0.47 of all the six categories combined. 
XLNet classifier reaches a validation loss about 1.5, F1-score about 0.21-0.25. 

Contributions:
Piyush Choudhari
Setting up the Google Cloud Instance deep learning image for GPU use and storage buckets
Re-trained the previous group’s TextCNN + skipgram model on Google cloud platform with change of parameters;
Preprocessed text data which brought the MSE down to 119 from previous 157
Used pre-trained GloVe embeddings to train the TextCNN model
Trained GloVe from scratch on the text from annual company reports
Scraped 2019 annual reports for two industries 
Data cleaning and spell correct to split the joined words in the report 
Categorized the bloomberg scores for the companies into score buckets 
Used a BERT classifier in Tensorflow to classify the companies into the score buckets created
Tried a few parameters to optimize the performance of the BERT model. 

Sareen Zhang
Improved and standardized the codes for scraping annual reports for all nine available industries
Scraped updated reports for five industries and cleaned data using word split
Improved the corpus construction code for data preprocessing and added more parameters for further analysis on the quality of data
Researched on climate risk keywords and other methods to measure risk
Obtained updated 2019 Bloomberg environment disclosure scores 

Luocen Wang
Scraped 2019 annual reports for five industries from websites.
Scraped the numeric data from sustainability reports, not continued later due to the difficulty of standardizing the data format of a large number of annual reports. 
Applied word-split codes on annual reports of two industries.
Researched on other environment scores and their standards. 
Constructed XLNet classifier through pre-trained models. 

Conclusion: After trying to optimize the TextCNN model to our best capacities and failing to achieve an MSE lower than 119, we see great potential in the classical methodology. One main difficulty presented in the classification model is the large amount of words in each company’s corpus. To train the model more in a more efficient and smarter way, further work on tokenization and encoding of the corpus is necessary. Besides, we believe that with further fine tuning, the model could perform better. 

Suggestions for further work:
Scraping more data: The main reason for the high loss value given by our BERT model is the dearth of company data. Reports other than annual reports could also prove to be useful in this task. The key is to obtain more companies that are mentioned in the bloomberg reports for ENVIRON_DISCLOSURE_SCORE so that merging the two datasets do not rule out most of the companies. Better keywords selection might also help with the quality of data extracted. Analysis on the data for each sector and those with extreme scores could be beneficial to understanding the actual texts and what is missing. 
Optimizing the TextCNN+GloVe model: Removing stopwords and changing parameters significantly increased TextCNN + Skipgram model. Thus we think doing similar things to GloVe might lead to an increase in performance as well. 
Optimizing the BERT model: We feel that the BERT model has a high potential to accurately classify the companies into score brackets. With further tuning and grid-search, we believe that the loss could be reduced significantly.
Optimize the XLNet model: XLNet model is an autoregressive model constructed based on the Bert model. Modifying the Tokenizer creation and feature extraction part in codes, as well as changing the parameters, may improve the model performance a lot. 
Try more models like Albert, Roberta, Longformer, and GPT-2: Based on the result from the Bert model, there is still abundant space to improve the performance. One major difficulty in training and fine-tuning the model is the enormous amount of corpus words for each company. Albert, Roberta, and Longformer models are Bert-related models. GPT-2 is a popular NLP model currently. 
Obtaining more standardized environment scores except Bloomberg ESG: on the Internet, there are many institutions and websites working on the environment and social scores. Websites like https://www.csrhub.com/ also hold major discussions on companies performance over sustainability, besides, this website also revealed part of its measure standard over sustainability performance. In some other environment scores, a company’s contribution and influence in the local community is included as an important standard. Researching different measure standards allows a comprehensive understanding of companies’ influence on environments. 

