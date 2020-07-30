

import argparse

print('----pls enter keyword,artistname,imagepath----')
print('python3 MAWukiyoe.py --keyword 日本橋夜景')
print('python3 MAWukiyoe.py --artists_Name 広重')
print('python3 MAWukiyoe.py --Image_path the-great-wave-of-kanagawa-1831.jpeg')
print('----or----all of them like:---')
print('python3 MAWukiyoe.py --artists_Name 広重 --keyword 景 --Image_path xxxx.jpeg')
parser = argparse.ArgumentParser(description='pls enter keyword,artistname,imagepath')

parser.add_argument('--keyword', dest='title', type=str, default='', help='fuji')
parser.add_argument('--artists_Name', dest='artist', type=str, default='', help='Yoshitoshi Tsukioka')
parser.add_argument('--Image_path', dest='imagepath', type=str, default='', help='193008.jpg')
args = parser.parse_args()



from pysparnnRANK import searchResult
#datalist,labelist,title,name,imgpath
print('--------------------result-----------------------------')
for eachitem in searchResult('data/crossmodal_VGGlast.csv','data/Icadlname_crossmodal_VGGlast.csv',args.title,args.artist,args.imagepath):
    print(eachitem+'/n' )




'''
from pysparnnRANK import  cosine_sim
print(cosine_sim('data/crossmodalEffi.csv','data/ICADL_dicembedding.csv',args.title,args.artist,args.imagepath))

'''

