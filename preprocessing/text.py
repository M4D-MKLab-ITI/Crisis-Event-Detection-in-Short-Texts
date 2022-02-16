import nltk
import numpy as np
import tensorflow as tf


# text preprocessing pipelines
# TODO: further inspection and development of the preprocessing pipeline

"""
Pipeline used in paper
"""
def preprocessing(tw):
    tw = [nltk.re.sub(r"http\S+", "link", text) for text in tw]  # replacing links: <LINK>
    tw = [nltk.re.sub(r"@\S+", "tag", text) for text in tw]  # replacing tags: <TAG>
    tw = [nltk.re.sub(r'[0-9]+', " digits ", text) for text in tw]  # replacing tags: digit
    tw = [nltk.re.sub(r"[\'|\"]", " ", text) for text in tw]  # removing ' and "
    tw = [nltk.re.sub(r"\b\w\b", "", text) for text in tw]  # remove single character words
    text_to_list = tf.keras.preprocessing.text.text_to_word_sequence
    tw = [text_to_list(sentence) for sentence in tw]
    stopwords = set(nltk.corpus.stopwords.words('english'))
    tw = [[word for word in words if word not in stopwords] for words in tw]
    tw = [" ".join(tweet) for tweet in tw]
    return np.asarray(tw)


def preprocessing2(tw, labs):
    to_delete = []
    for idx, tweet in enumerate(tw):
        full_length = len(tw[idx])
        tw[idx] = nltk.re.sub(r'[^\x00-\x7f]', " ", tweet)
        if len(tw[idx]) / full_length < 0.8:
            to_delete.append(idx)
    tw = [nltk.re.sub(r"\b\w\b", "", text) for text in tw]  # remove single character words

    text_to_list = tf.keras.preprocessing.text.text_to_word_sequence
    tw = [text_to_list(sentence) for sentence in tw]
    stopwords = set(nltk.corpus.stopwords.words('english'))
    tw = [[word for word in words if word not in stopwords] for words in tw]
    tw = [" ".join(tweet) for tweet in tw]
    # remove \r in text
    tw = [nltk.re.sub(r"\r", " ", text) for text in tw]  # removing ' and "
    for idx, tweet in enumerate(tw):
        unique = set(tweet.strip().split())
        flag_word_counter = sum([1 if x in unique else 0 for x in ["rt", "tag", "link"]])
        if tw[idx].strip() == "" or (len(unique) <= 3 and flag_word_counter == len(unique) - 1):
            to_delete.append(idx)
    tw = np.delete(tw, list(set(to_delete)))
    labs = np.delete(labs, list(set(to_delete)), axis=0)
    return np.asarray(tw), labs
