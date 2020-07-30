import numpy as np
import PretrainBertprocessing


class queryProcessing:
    def __init__(self, imagepath='', title='', artist=''):
        self.image_path = imagepath
        self.title = title
        self.artist = artist

    def conectquery(self):
        outputVec = []
        if len(self.title[0].replace(' ', '')) > 0:
            titVec = PretrainBertprocessing.query2BertVec().textsEmbedding(self.title)
        else:
            titVec = np.zeros((1, 768))
        if len(self.artist[0].replace(' ', '')) > 0:
            artistVec = PretrainBertprocessing.query2BertVec().textsEmbedding(self.artist)
        else:
            artistVec = np.zeros((1, 768))
        if len(str(self.image_path).replace(' ', '')) > 0:
            from Effi_extractors import pretreatment
            imaVec = pretreatment(self.image_path)
            #from HogExtracter import hogpretreatment

            #imaVec = hogpretreatment(self.image_path)


        else:
            imaVec = np.zeros((4096,))

        outputVec = np.concatenate((titVec[0], artistVec[0], imaVec), axis=0)
        #print(np.shape(outputVec))
        #for each in outputVec:
            #print(imaVec)


        return outputVec

