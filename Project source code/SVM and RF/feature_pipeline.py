import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from textblob import TextBlob

import lda


# TODO: use only polarity and ignore subjectivity
# feature extractor for review level sentiment
class TextSentiment(BaseEstimator, TransformerMixin):
    """
        Returns list of [polarity, subjectivity] for each review
    """

    def fit(self, x, y=None):
        """
        fit function which is applied on training data
        it inturn calls transform to transform input
        :param x: all the x as a list
        :param y: all of y as list
        :return: self (performs transform for train data x)
        """
        return self

    def transform(self, reviews):
        """
        converts input to a number representation for use with ML algos
        :param reviews: test reviews list
        :return: list of converted reviews
        """
        # create an array with polarity,subjectivity
        textblob_output = []
        # for each review
        for review in reviews:
            # instantiate text blob
            tb = TextBlob(review)
            # create a temporary array
            temp = []
            # append polarity and subjectivity
            temp.append(tb.sentiment.polarity)
            temp.append(tb.sentiment.subjectivity)
            # append it to output list
            textblob_output.append(temp)
        # return converted output
        return textblob_output


class LDATopics(BaseEstimator, TransformerMixin):
    """
    converts reviews into list of topics
    """
    def __init__(self, topics_number=None, lda_instance=None):
        """
        initializer for the instance
        :param lda_instance: old model from disk
        :return: None
        """
        # load if instance is passed
        self.lda_instance = lda_instance
        # set number of topics
        if topics_number is None:
            self.topics_number = 10
        else:
            self.topics_number = topics_number

    def fit(self, x, y=None):
        """
        Fit function to create an lda model
        :param x:
        :param y:
        :param topics:
        :return:
        """
        if self.lda_instance is None:
            self.lda_instance = lda.perform_lda(x, self.topics_number)
        return self

    def transform(self, reviews):
        """

        :param reviews:
        :param topics:
        :return:
        """
        # create an array with topic distribution
        topic_dist_list = []

        # generate sistribution
        topic_dist_list = lda.generate_topic_dist_matrix(self.lda_instance,
                                                         self.topics_number,
                                                         reviews, topic_dist_list)
        cols = []
        # add columns to data
        for i in xrange(1, self.topics_number + 1):
            cols.append("Topic" + str(i))
        # create a dataframe
        topic_dist_df = pd.DataFrame(topic_dist_list, columns=cols)
        features = list(topic_dist_df.columns[:self.topics_number])
        x_train = topic_dist_df[features]
        # convert into list and send back
        return map(list, x_train.values)
