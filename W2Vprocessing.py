import pickle
from gensim.models.word2vec import Word2Vec


class queryTow2v:
    def __init__(self, dic_path, model_path):
        with open(dic_path, mode="rb") as f:
            self.word2sid = pickle.load(f)
        self.sid2word = {v:k for k, v in self.word2sid.items()}
        self.model = Word2Vec.load(model_path)

    def get_vector(self, token):

        sid = self.word2sid[token]
        word_vector = self.model.wv[sid]
        print('-- {} --'.format(token))
        print(word_vector)

    def get_similar_tokens(self, token, topn=5):

        sid = self.word2sid[token]
        similar_tokens = self.model.wv.most_similar(positive=[sid], topn=topn)
        print('-- {} --'.format(token))
        for sid, similarity in similar_tokens:
            token = self.sid2word[sid]
            print("{}: {:.2f}".format(token, similarity))
    def get_result(self,size = 768,min_count = 5,window = 20,sg = 1,mode='vector',sample = "立命"):
        dirname = "size{}-min_count{}-window{}-sg{}".format(size, min_count, window, sg)
        self.dic_path = "dictionary.pickle"
        self.model_path = "./model/" + dirname + "/wikipedia.model"
        if mode=='vector':
            return self.get_vector(sample)
        else:return self.get_similar_tokens(sample)


