
from multilabel_knn import multilabel_knn, binom_multilabel_kNN, evaluation
import joblib
from sklearn.metrics import hamming_loss
import pickle
import scipy.sparse
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
import re
import string

class ModelCategorization:
    def __init__(self, path_to_model, path_to_categories, path_to_vectorizer):
        self.path_to_model = path_to_model
        self.path_to_categories = path_to_categories
        self.path_to_vectorizer = path_to_vectorizer
        self.model = None
        self.categories = None
        self.vectorizer = None
        self.instantiate_model()
    
    def instantiate_model(self):
        self.model = joblib.load('/Users/jitcad/Documents/NLP_problem/files/model_trained')

        self.categories = pickle.load(open(self.path_to_categories, 'rb'))
        self.vectorizer = joblib.load(self.path_to_vectorizer)

    def clean_text(self, text):
        '''Make text lowercase, remove links, remove punctuation, remove end line, 
        remove words containing special char, lemmatize

        '''
        
        lemmatizer = WordNetLemmatizer()
    #     words_english = set(nltk.corpus.words.words())
        
        text = str(text).lower()
        text = re.sub('https?://\S+|www\.\S+', '', text)
        text = re.sub('<.*?>+', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
        text = re.sub('\n', ' ', text)
    #     text = re.sub('[1-9]', ' ', text)
        text = ' '.join([ word for word in word_tokenize(text) if word == re.sub(r'[^a-zA-Z]', '', word)])
    #     text = ' '.join(word for word in word_tokenize(text) if word in words_english)
        text = ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(text)])
        return text


    def get_categories(self, one_hot):
        """Get name of categories based on one hot encode vector result
        Args:
            one_hot (array): one hot vector
    
        Returns:
            list (str): name of categories 
        """
        result = []

        for idx, i in enumerate(one_hot):
            if i:
                result.append(self.categories[idx])
        return result

    def get_categories_with_idx(self, idxs):
        """Get name of categories based indexes
        Args:
            idxs (array): indexes from where to get categories
    
        Returns:
            list (str): name of categories 
        """
        result = []
        for idx in idxs:
            result.append(self.categories[idx])

        return result

    def get_top_5_categories(self, text):
        """Get top five categories for infered text
        Args:
            pred (array): predicted categories for infered texts

        Returns:
            gt_classes (array of str): array of ground truth categories for every  text
            res_classes (array of str): array of categories for every infered text
        """
        pred = self.infer_text(text)
        top_5 = np.flip(np.argsort(pred))[:5]
        # print(top_5)
        res_categories = self.get_categories_with_idx(top_5)
    
        return res_categories

    def infer_text(self, text):
        """Infere text with model
        Args:
            text (str): the text to be infered

        Returns:
            Y_prob (array of str): array of probabilities for every category of the infered text
        """
        text = self.clean_text(text)
        tfidf = self.vectorizer.transform([text]).toarray()
        Y_prob = self.model.predict(tfidf, return_prob = True)
        Y_prob = scipy.sparse.csr_matrix.toarray(Y_prob[1])[0]
        return Y_prob
    
    def infer_array_of_texts(self, texts, with_probabilities=False):
        """Infere text with model
        Args:
            text (str): the text to be infered

        Returns:
            Y_prob (array of str): array of probabilities for every category of the infered text
                                (if with_probabilities is True) or one hot vector if not
        """
        if with_probabilities:
            Y_prob = self.model.predict(texts, return_prob = True)
            Y_prob = scipy.sparse.csr_matrix.toarray(Y_prob[1])
        else:
            Y_prob = self.model.predict(texts)

        return Y_prob


# test_X = pickle.load(open('test_X.pkl', 'rb'))
# test_Y = pickle.load(open('test_Y.pkl', 'rb'))
# train_X = pickle.load(open('train_X.pkl', 'rb'))
# train_Y = pickle.load(open('train_Y.pkl', 'rb'))

model = Model('/Users/jitcad/Documents/NLP_problem/files/model_trained', '/Users/jitcad/Documents/NLP_problem/files/categories.pkl', '/Users/jitcad/Documents/NLP_problem/files/vectorizer.pkl')
with open('/Users/jitcad/Documents/NLP_problem/text.txt') as f:
    text = f.read()
print(model.get_top_5_categories(text))