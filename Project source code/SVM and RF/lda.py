import re
from gensim import corpora, models
from nltk.corpus import stopwords


# get all stopwords
stoplist = set(stopwords.words("english"))


# methods for helping lda
def perform_lda(train_reviews, topics_number):
    """
    train lda model with given data and number of topics
    :param train_reviews:
    :param topics_number:
    :return: lda trained instance
    """
    corpus = []
    for review in train_reviews:
        # Remove punctuations
        review = re.sub(r'[^a-zA-Z]', ' ', review)
        # To lowercase
        review = review.lower()
        # Remove stop words
        texts = [word for word in review.lower().split() if word not in stoplist]
        try:
            corpus.append(texts)
        except:
            pass

    # Build dictionary
    dictionary = corpora.Dictionary(corpus)
    dictionary.save('restaurant_reviews.dict')

    # Build vectorized corpus
    corpus_2 = [dictionary.doc2bow(text) for text in corpus]

    # compile lda model
    lda = models.LdaModel(corpus_2, num_topics=topics_number, id2word=dictionary)

    # return instance of lda
    return lda


def generate_topic_dist_matrix(lda, numTopics, corpus, all_dist):
    """
    generate topic distribution of each document
    :param lda: instance of lda
    :param numTopics: number of topics to be used
    :param corpus: corpus
    :param all_dist: distribution keep sake
    :return: return matrix of reviews and their distributions
    """
    # list of topic distributions
    topic_dist = [0] * numTopics
    # load dictionary
    dictionary = corpora.Dictionary.load("restaurant_reviews.dict")
    # for each review generate distribution
    for doc in corpus:
        vec = dictionary.doc2bow(doc.lower().split())
        output = lda[vec]
        highest_prob = 0
        highest_topic = 0
        temp = [0] * numTopics
        # List to keep track of topic distribution for each document
        for topic in output:
            this_topic, this_prob = topic
            temp[this_topic] = this_prob
            if this_prob > highest_prob:
                highest_prob = this_prob
                highest_topic = this_topic

        all_dist.append(temp)
        topic_dist[highest_topic] += 1
    # return dictionary
    return all_dist