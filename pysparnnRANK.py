import pysparnn.cluster_index as ci
import numpy as np
import queryEmbedding
import pandas as pd
import csv
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
def getNamelist(dataPath):

    csvreader = pd.read_csv(dataPath, header=None)

    final_list = csvreader.values.tolist()
    return final_list
def searchResult(datalist,labelist,title,name,imgpath):
    querylist=[]
    query=queryEmbedding.queryProcessing()
    query.image_path=imgpath
    query.artist=[name]
    query.title=[title]
    w=query.conectquery()
    #querylist.append(query.conectquery())
    OPlist=[]

        

    record=[]
    
    for each in getNamelist(labelist):
        if 'http' in str(each[3]):
            record.append('[ARC Database]'+str(each[1])+' by:'+str(each[2])+'————info.'+str(each[3]))
        else:record.append('[WikiArt]you can check ID:'+str(each[0])+':'+str(each[1])+' by:'+str(each[2])+'in Wikiart,or search by  using image from wikiart or by using keywords:'+str(each[3]))
    
    if len(str(imgpath).replace(' ', '')) <1:
        querylist.append(w[:1536])
        cp = ci.MultiClusterIndex(np.array(getNamelist(datalist))[:, :1536], record)
        wodTP3=cp.search(querylist, k=5, k_clusters=50, return_distance=False)[0]
        for each in wodTP3:
            OPlist.append(each)
    elif len(str(title).replace(' ', '')) <1 and len(str(name).replace(' ', '')) <1:
        querylist.append(w[1536:])
        cp = ci.MultiClusterIndex(np.array(getNamelist(datalist))[:, 1536:], record)
        imgTP3=cp.search(querylist, k=5, k_clusters=50, return_distance=False)[0]
        for each in imgTP3:
            OPlist.append(each)
        else:        
            cp = ci.MultiClusterIndex(np.array(getNamelist(datalist)), record)
            imgTP3=cp.search(w, k=5, k_clusters=50, return_distance=False)[0]
            for each in imgTP3:
                OPlist.append(each)
    #print(getNamelist(datalist)[record.index(r)])
    return OPlist


def cosine_sim(datalist,labelist,title,name,imgpath):

    query=queryEmbedding.queryProcessing()
    query.image_path=imgpath
    query.artist=[name]
    query.title=[title]
    record=[]
    for each in getNamelist(labelist):
        record.append(str(each[0])+':'+str(each[1])+' by:'+str(each[2]))
    querylist = np.tile(query.conectquery(), len(record))
    w=cosine_similarity(np.array(getNamelist(datalist)).reshape(1998, -1),np.array(querylist).reshape(1998, -1))
    return record[np.argwhere(w == np.max(w))[0][0]]

import pickle
def readpickle(path):
    with open(path, mode="rb") as f:
            return pickle.load(f)

imgdic=readpickle('imgrink')
def datasetmaker(title,name,id):
    print(title)
    query=queryEmbedding.queryProcessing()
    query.image_path=imgdic[id]
    query.artist=[name]
    query.title=[title]
    return query.conectquery()
