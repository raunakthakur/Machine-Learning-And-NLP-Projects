
## Natural Language Processing (NLP) - Twitter sentiment Analysis


### Context
This is the sentiment140 dataset. It contains 1,600,000 tweets extracted using the twitter api . The tweets have been annotated (0 = negative, 4 = positive) and they can be used to detect sentiment .

### Data Information:
It contains the following 6 fields:<br>

target: the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)<br>
ids: The id of the tweet ( 2087)<br>
date: the date of the tweet (Sat May 16 23:58:44 UTC 2009)<br>
flag: The query (lyx). If there is no query, then this value is NO_QUERY.<br>
user: the user that tweeted (robotickilldozr)<br>
text: the text of the tweet (Lyx is cool)<br>

### Libraries
- pandas
- matplotlib
- seaborn
- scikit-learn
- re
- spacy
- sklearn
- wordcloud 
- plotly
- BeautifulSoup
- warnings

### Algorithms

- SGD Classifier
- LogisticRegression
- LogisticRegressionCV
- LinearSVC
- RandomForestClassifier

### Model Evalutaion -<br>
TFIDF Vectorizer
1. Stochastic Gradient Descent - 0.684
2. Logistic Regression - 0.722
3. Logistic Regression-CV - 0.713
4. Support vector machine - 0.691
5. Random Forest classifier - 0.651

### Project Utility –
Catching Trending Buzz Words in Twitter through Word cloud for Advertisement , sentiments acknowledgement and strategy building . sentiment prediction about specific Named Entity Recognition and come out with unique strategy to improve sentiments regarding the same.  
