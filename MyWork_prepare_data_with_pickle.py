
#coding:utf-8

# In[33]:

import pickle
import io
import sys
import re

title_file = open('data/Wechat_title.parsed.txt', encoding='utf8')
content_file = open('data/Wechat_content.parsed.txt', encoding='utf8')

title_line = title_file.readlines()
content_line = content_file.readlines()

#print(len(title_line))
#print(len(content_line))

pkl_file = open('data.pkl', 'wb')
data  = {
    'title':title_line,
    'content':content_line,
    'keyword':None
}

pickle.dump(data, pkl_file)
pkl_file.close()


# In[34]:

import pickle
pkl_file = open('data.pkl', 'rb')

data = pickle.load(pkl_file)

pkl_file.close()

sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='gb18030')
#print("title: \n", data['title'])
#print("content:\n", data['content'])


# In[1]:

# Code here 

def load_wordembedding(file_dir):
    
    file = open(file_dir, encoding='utf8')
    line = file.readline()
    words = []
    vectors = []
    i = 0
    while line:
        if i == 300000:
            break
        temp_tokens = re.split(",|\\n", line) # line.split(",")
        if len(temp_tokens) != 202:
            print("error length: %d\t"%len(temp_tokens), temp_tokens[0:3])
            line = file.readline()
            continue
        elif i%1000 == 0:
            print("%d\n"%i)
        words.append(temp_tokens[0])
        vectors.append([float(x) for x in temp_tokens[1:201]])
        line = file.readline()
        i += 1
    return np.array(words), np.array(vectors)


# In[ ]:

def test_wordembedding():
    words, vectors = load_wordembedding("data/word_vector.csv")
    print(words.shape, vectors.shape)
    assert words.shape== (9846,)
    assert vectors.shape == (9846,200)
    


# In[ ]:

#code here: test the data we have so far.  
# test_wordembedding()
#print(title_words[0])
#print(content_words[0])
words, vectors = load_wordembedding("data/vector.bin.csv")
print(words.shape, vectors.shape)

