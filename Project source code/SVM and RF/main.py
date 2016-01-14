import pickle

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer

from feature_pipeline import TextSentiment, LDATopics
from preprocess import process_reviews
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# read datasets
businessData = pd.read_csv('data/yelp_academic_dataset_business.csv', dtype=unicode)
reviewData = pd.read_csv('data/yelp_academic_dataset_review.csv')
userData = pd.read_csv('data/yelp_academic_dataset_user.csv')


# Merge dataframes using pandas
reviewUserdata = reviewData.merge(userData,
                                     left_on='user_id',
                                     right_on='user_id',
                                     how='outer',
                                     suffixes=('_review', '_user'))

businessReviewuserdata = reviewUserdata.merge(businessData,
                                                   left_on='business_id',
                                                   right_on='business_id',
                                                   how='outer',
                                                   suffixes=('_reviewuser', '_business'))
# rename the columns
businessReviewuserdata = businessReviewuserdata.rename(columns={'name_reviewuser': 'name_user',
                                                                      'review_count_reviewuser': 'review_count_user',
                                                                      'stars_reviewuser': 'stars_review'})

cols = businessReviewuserdata.columns
cols = cols.map(lambda x: x.replace(' ', '_').lower() if isinstance(x, (str, unicode)) else x)
businessReviewuserdata.columns = cols

# Extract columns from dataframe
firstFewColumns = businessReviewuserdata.ix[:, ['categories', 'text', 'stars_review', 'business_id']]
rDF = firstFewColumns[firstFewColumns['categories'].str.contains('Restaurants')]

print 'No of reviews (Before dropping NaaN): ', len(rDF.index)
rDF = rDF[np.isfinite(rDF['stars_review'])]
print 'No of reviews (After dropping NaaN): ', len(rDF.index)

# rDF['predicted_rating'] = round(sum(rDF.stars_review)/len(rDF.index))

# Stop words from corpus
stoplist = set(stopwords.words("english"))
numTopics = 15
t = rDF.dropna(how='all')
minReviewLen = 25
maxReviewLen = 100

print "Number of reviews selected:", len(t[t.text.str.len() > minReviewLen][t.text.str.len() < maxReviewLen])
rDF = t[t.text.str.len() > minReviewLen][t.text.str.len() < maxReviewLen]

## Grouping the review based on star rating
starsGroup = rDF.groupby('stars_review')

oneStarText = starsGroup.get_group(1.0)['text']
twoStarsText = starsGroup.get_group(2.0)['text']
threeStarsText = starsGroup.get_group(3.0)['text']
fourStarsText = starsGroup.get_group(4.0)['text']
fiveStarsText = starsGroup.get_group(5.0)['text']

oneStarLabels = [1.0] * len(oneStarText)
twoStarLabels = [2.0] * len(twoStarsText)
threeStarLabels = [3.0] * len(threeStarsText)
fourStarLabels = [4.0] * len(fourStarsText)
fiveStarLabels = [5.0] * len(fiveStarsText)

## Split test and train reviews
oneStarText_train, oneStarText_test, oneStarLabels_train, oneStarLabels_test = train_test_split(
    oneStarText, oneStarLabels, test_size=0.10)
twoStarsText_train, twoStarsText_test, twoStarLabels_train, twoStarLabels_test = train_test_split(
    twoStarsText, twoStarLabels, test_size=0.10)
threeStarsText_train, threeStarsText_test, threeStarLabels_train, threeStarLabels_test = train_test_split(
    threeStarsText, threeStarLabels, test_size=0.10)
fourStarsText_train, fourStarsText_test, fourStarLabels_train, fourStarLabels_test = train_test_split(
    fourStarsText, fourStarLabels, test_size=0.10)
fiveStarsText_train, fiveStarsText_test, fiveStarLabels_train, fiveStarLabels_test = train_test_split(
    fiveStarsText, fiveStarLabels, test_size=0.10)

##Pre processing the review text
# Process the reviews
corpus_5stars = process_reviews(fiveStarsText_train)
corpus_4stars = process_reviews(fourStarsText_train)
corpus_3stars = process_reviews(threeStarsText_train)
corpus_2stars = process_reviews(twoStarsText_train)
corpus_1stars = process_reviews(oneStarText_train)


fiveFourText_train = np.append(corpus_5stars, corpus_4stars)
fiveFourThreeText_train = np.append(fiveFourText_train, corpus_3stars)
fiveFourThreeTwoText_train = np.append(fiveFourThreeText_train, corpus_2stars)
allText_train = np.append(fiveFourThreeTwoText_train, corpus_1stars)


## Create training labels
fiveFour_label = np.append(fiveStarLabels_train, fourStarLabels_train)
fiveFourThree_label = np.append(fiveFour_label, threeStarLabels_train)
fiveFourThreeTwo_label = np.append(fiveFourThree_label, twoStarLabels_train)
allLabel_train = np.append(fiveFourThreeTwo_label, oneStarLabels_train)


# Process the test reviews
corpus_5stars = process_reviews(fiveStarsText_test)
corpus_4stars = process_reviews(fourStarsText_test)
corpus_3stars = process_reviews(threeStarsText_test)
corpus_2stars = process_reviews(twoStarsText_test)
corpus_1stars = process_reviews(oneStarText_test)



fiveFourText_test = np.append(corpus_5stars, corpus_4stars)
fiveFourThreeText_test = np.append(fiveFourText_test, corpus_3stars)
fiveFourThreeTwoText_test = np.append(fiveFourThreeText_test, corpus_2stars)
allText_test = np.append(fiveFourThreeTwoText_test, corpus_1stars)


## Create test labels
fiveFour_label = np.append(fiveStarLabels_test, fourStarLabels_test)
fiveFourThree_label = np.append(fiveFour_label, threeStarLabels_test)
fiveFourThreeTwo_label = np.append(fiveFourThree_label, twoStarLabels_test)
allLabel_test = np.append(fiveFourThreeTwo_label, oneStarLabels_test)


# Generate tf-idf vectors
vectorizer = TfidfVectorizer()
senti = TextSentiment()
lda = LDATopics()

#Baseline model 1
combined_features = FeatureUnion([("tfidf", vectorizer)])

#Baseline model 2
#combined_features = FeatureUnion([("sentiment", senti)])

#Baseline model 3
#combined_features = FeatureUnion([("LDA", lda)])

# Combined model
#combined_features = FeatureUnion([("tfidf", vectorizer), ("sentiment", senti), ("LDA", lda)])

# SVM
rf = svm.SVC(class_weight="balanced", kernel="linear", gamma="auto")

# Random Forest
#rf = RandomForestClassifier(n_estimators=100, n_jobs=2)

pipeline = Pipeline([("features", combined_features),("Imputer", Imputer(missing_values='NaN',
                strategy="mean",
                axis=1)), ("rf", rf)])
pipeline.fit(allText_train, allLabel_train)
preds = pipeline.predict(allText_test)

print "\n"
print "Confusion matrix: Rows and Columns in the order 1, 2, 3, 4, 5"
print "\n"
print confusion_matrix(allLabel_test, preds, labels=[1.0, 2.0, 3.0,4.0,5.0])
