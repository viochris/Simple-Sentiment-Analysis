from sklearn.feature_extraction.text import CountVectorizer
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

def get_top_n_words_en(corpus, n, ngram_range):
    vec = CountVectorizer(ngram_range=ngram_range, stop_words='english').fit(corpus)
    bag = vec.transform(corpus)
    sum_words = bag.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    return sorted(words_freq, key=lambda x: x[1], reverse=True)[:n]


def get_top_n_words_id(corpus, n, ngram_range):
    stopword = stopwords.words('indonesian')
    if "luar" in stopword or "biasa" in stopword:
        stopword.remove("luar")
        stopword.remove("biasa")
    
    vec = CountVectorizer(ngram_range=ngram_range, stop_words=stopword).fit(corpus)
    bag = vec.transform(corpus)
    sum_words = bag.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    return sorted(words_freq, key=lambda x: x[1], reverse=True)[:n]

def convert_for_download(df):

    return df.to_csv(index=False).encode("utf-8")


