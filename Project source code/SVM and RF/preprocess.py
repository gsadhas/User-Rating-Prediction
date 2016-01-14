import re

from nltk.corpus import stopwords

def process_reviews(datasetReview):
    cleanDataset = []
    for review in datasetReview:
        # Remove punctuations from review text
        review = re.sub(r'[^a-zA-Z]', ' ', review)
        # Convert to lowercase
        review = review.lower()
        # Remove stop words from text
        stoplist = set(stopwords.words("english"))
        texts = [word for word in review.lower().split() if word not in stoplist]
        try:
            cleanDataset.append(' '.join(texts))
        except:
            pass
    return cleanDataset