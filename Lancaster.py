#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Having reviewed two NLP library(NLTK,spaCy), I chose NLTK library. Since i do ont intend to adpot td-idf and word vector, 
# instead count the numebr of matching tagged words. And NLTK performs better in sentence tokenization.
# NLTK enbales users to adpot classical rule-based approach and could be more explainable and transparent for analysis


# In[36]:


from urllib.request import urlopen
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
import heapq
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')


# In[37]:


f = urlopen('https://www.bbc.com/news/business-52905265')
soup = BeautifulSoup(f,'html.parser')


# In[42]:


p_list = [soup.title.string] + soup.find_all('p') 
tokenizer = nltk.RegexpTokenizer(r'\w+')
st = []
title = set()
POS = set()

for i,p in enumerate(p_list):
    if p.string != None:
        word_list = tokenizer.tokenize(p.string)
        w = [word for word in word_list if word not in stopwords.words('english')]
        pos_tag = nltk.pos_tag(w)
        chunk = nltk.ne_chunk(pos_tag)
        NE = [" ".join(w for w ,t in ele) for ele in chunk if isinstance(ele,nltk.Tree)]
        if i == 0:
            for ele in chunk:
                if not isinstance(ele,nltk.Tree) and ele[1] == 'NNS':
                    title.add(ele[0])
                elif not isinstance(ele,nltk.Tree):
                    POS.add(ele)
            if NE:
                title |= set(NE)
        else:
#             calculate the number of matching NE and POS tagged words            
            m = 0
            for c in NE:
                if c in title:
                    m += 1   
            for ele in chunk:
                if not isinstance(ele,nltk.Tree) and (ele[0] in title or ele in POS):
                    m += 1
#              sorting based on matching number increasingly and index from article decreasingly, so that we can add top ranking sentence with same mathing number in original order       
            if m:
                heapq.heappush(st,[m,-i,p.string])

res = []
total_words = 0
for _,_,s in st[::-1]:
    total_words += len(s.string)
    res.append(s.string)
    if total_words > 140:
        break

print(','.join(res))





