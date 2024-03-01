from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from numpy.linalg import norm
from nltk.corpus import stopwords
import string

class TFIDFUtil:

    def clean_tuple(self, t):
        return ' '.join([str(item) for item in t])

    def clean_doc(self, doc):
        # split into tokens by white space
        tokens = doc.split()
        # remove punctuation from each token
        table = str.maketrans('', '', string.punctuation)
        tokens = [w.translate(table) for w in tokens]
        # remove remaining tokens that are not alphabetic
        tokens = [word for word in tokens if word.isalpha()]
        # filter out stop words
        stop_words = set(stopwords.words('english'))
        tokens = [w for w in tokens if not w in stop_words]
        # filter out short tokens
        tokens = [word for word in tokens if len(word) > 2]
        return tokens

    def tfidf_similarity(self, s1, s2):
        def add_space(s):
            return ' '.join(s)

        # 将字中间加入空格
        s1, s2 = add_space(s1), add_space(s2)
        # 转化为TF矩阵
        cv = TfidfVectorizer(tokenizer=lambda s: s.split())
        corpus = [s1, s2]
        vectors = cv.fit_transform(corpus).toarray()
        # 计算TF系数
        norm_0 = norm(vectors[0])
        norm_1 = norm(vectors[1])
        if norm_0 == 0 or norm_1 == 0:
            return 0
        return np.dot(vectors[0], vectors[1]) / (norm(vectors[0]) * norm(vectors[1]))

    def get_tfidf_similarity(self, clean_list):
        total = 0.0
        num = 0.0
        for i in clean_list:
            for j in clean_list:
                if i != j:
                    num += 1
                    total += self.tfidf_similarity(i, j)
        if num == 0:
            return 0
        return total/num

    def calculate_similarity(self, documents):
        clean_documents = [self.clean_doc(self.clean_tuple(doc)) if isinstance(doc, tuple) else self.clean_doc(doc) for doc in documents]
        avg_similarity = self.get_tfidf_similarity(clean_documents)
        return avg_similarity

# # 使用示例
# tfidf_util = TFIDFUtil()
# documents = [...]  # 填充文档列表
# similarity = tfidf_util.calculate_similarity(documents)
# print(similarity)
