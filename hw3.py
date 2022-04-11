"""
Goal: Build an index and a basic retrieval component
By basic retrieval component; we mean that at this point you just need to be able to query your index
for links (The query can be as simple as single word at this point).
These links do not need to be accurate/ranked. We will cover ranking in the next milestone.
At least the following queries should be used to test your retrieval:
1 - Informatics
2 - Mondego
3 - Irvine 

your index should store the TF-IDF of every term/document. 
Words in title, bold and heading (h1, h2, h3) tags are more important than the other words. 
You should store meta-data about their importance to be used later in the retrieval phase.

You can either use a database to store your index (MongoDB)

------------------------------------------Deliverables---------------------------------------

Submit a report (pdf) in Canvas with the following content: 
    1. A table with assorted numbers pertaining to your index. It should have:
        - at least the number of documents --> corpus has around 35k
        - the number of [unique] words-- > len(set(tokens))
        - total size (in KB) of your index on disk. ------> TO DO
    2. Number of URLs retrieved for each of the queries above and listing the first 20 URLS for each
       query 


----------------------------------------Evaluation criteria----------------------------------

● Was the report submitted on time?
● Are the reported numbers plausible?
● Are the reported URLs plausible?
 


"""
from doctest import DONT_ACCEPT_TRUE_FOR_1
import nltk #tokenizer
import pickle
import json
from nltk.stem import WordNetLemmatizer
from sympy import E
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
import os
import re
import sys
import math
import numpy as np
from collections import defaultdict

stop_words=set()
lemmatizer = WordNetLemmatizer()
unique_words = set()
doc_count=0
result_count=0


################################# TOKEN PROCESSING #######################################
#collect stop words
def get_stopword():
    stop_word_file = open('stop_word.txt', 'r')
    stop_word_list = stop_word_file.readlines()
    for stop_word in stop_word_list:
        stop_words.add(stop_word.strip())
    stop_word_file.close

#lemmatizer from word -> noun
def lemmatize(token):
    return lemmatizer.lemmatize(token, pos="n")

#################################### CREATE INDEX ########################################
#tokenize content of all files
def contentFromFile(folder,file):
    with open("./WEBPAGES_RAW/"+str(folder)+"/"+str(file),"r", encoding='utf-8') as curFile:
        content=curFile.read()

        #use regex to pick out <h1> ----> <h6> <b> <strong>:
        headers=[]
        headers+=(re.findall(r"<h1>[^<>]*<\/h1>",content))
        headers+=(re.findall(r"<h2>[^<>]*<\/h2>",content))
        headers+=(re.findall(r"<h3>[^<>]*<\/h3>",content))
        headers+=(re.findall(r"<h4>[^<>]*<\/h4>",content))
        headers+=(re.findall(r"<h5>[^<>]*<\/h5>",content))
        headers+=(re.findall(r"<h6>[^<>]*<\/h6>",content))
        headers+=(re.findall(r"<b>[^<>]*<\/b>",content))
        headers+=(re.findall(r"<strong>[^<>]*<\/strong>",content))
        #for header in headers: #-------> TEST
        #    print(header)
        
        #modify headers to remove <...> & </...>
        for i in range(len(headers)):
            headers[i]=re.sub(r"<[^<>]*>"," ",headers[i])
        #for header in headers: #-------> TEST
        #    print(header)
        
        
        content=re.sub(r"<script>[^<>]*<\/script>"," ",content) 
        content=re.sub(r"<style>[^<>]*<\/style>"," ",content)
        content=re.sub(r"<[^<>]*>"," ",content)

        result=nltk.word_tokenize(content)
        for header in headers:
            #print(nltk.word_tokenize(header))#-------> TEST
            result+=(nltk.word_tokenize(header)*9) #-----------> header is more important
    return result


#calculate frequencies 
"""
your index should store the TF-IDF of every term/document. 
Words in title, bold and heading (h1, h2, h3) tags are more important than the other words. 
You should store meta-data about their importance to be used later in the retrieval phase.
"""
def tokensToDictionary(tokens):
    dicta =defaultdict(int)
    for token in tokens:
        unique_words.add(token) #add to set of unique words
        token=lemmatize(token)
        if token not in stop_words:
            dicta[token]+=1
    for token in dicta:#-------------> TF Score & log frequency weight
        dicta[token]/=len(tokens)
        dicta[token]=1+math.log(dicta[token],10)
    return dicta


#add it to index as {token:[[file_id,frequency],...]}   ----->   file_id = "folder_number/file_number"
def dictionaryToIndex():
    index={}
    folder=0
    global doc_count
    doc_count=0
    while True:
        try:
            with open("./WEBPAGES_RAW/"+str(folder)+"/0","r", encoding='utf-8'):
                file=0
                while True:
                    try:
                        #if folder/file is not yet processed in disk index
                        for token, frequency in tokensToDictionary(contentFromFile(folder,file)).items():
                            if token not in index:
                                index[token]=[[str(folder)+"/"+str(file),frequency]]
                            else:
                                index[token].append([str(folder)+"/"+str(file),frequency])
                        print("{}/{} processed".format(folder,file))
                    except FileNotFoundError: break
                    doc_count+=1
                    file+=1
        except FileNotFoundError: break
        folder+=1
        #if folder==2: break
    #IDF = log((total # of document) / (number of docs containing token))
    for token in index:
        IDF=math.log(doc_count/len(index[token]),10)
        for i in range(len(index[token])):#-----------> TF-IDF
            index[token][i][1]*=IDF
        #print("TOKEN :   " + token)
        #print(index[token])
    #save index in form of binary to txt
    with open("index.pickle","wb") as pickled_index:
        pickle.dump(index,pickled_index)
    

    #returnindex
    

#search with query
def query(user_input,index):
    modified_input = lemmatize(user_input.lower().strip())
    if modified_input in index:
        return index[modified_input]
    else:
        return []

##################################### DISPLAY FOR UI/PDF#########################################
def displayQueryResult(user_input):
    
    #calculate index if not yet calculated
    try:
        with open("index.pickle","rb") as pickled_index:
            index=pickle.load(pickled_index)
    except FileNotFoundError:
        dictionaryToIndex()
        with open("index.pickle","rb") as pickled_index:
            index=pickle.load(pickled_index)

    #calculate docs' score
    """
    scores=defaultdict(int)
    for term in set(nltk.word_tokenize(user_input)): #split into terms
        print("scoring for: ",term)
        for i in query(term,index): #calculate tf-idf for docs containing current term
            scores[i[0]]+=i[1] #add up total scoring for query (term + term + ...)
    query_result=[]
    for i in scores:
        query_result.append([i,scores[i]])
    query_result=sorted(query_result, key = lambda x: -x[1])
    """
    """
    1/ collect all terms from query ---> we get N dimensions (N=# of unique query term) /////call query()
    2/ collect all docs containing more than 1 of the terms in query /// call query(each_term) --> list of doc containing that term
    3/ buils matrix AKA list of vectors //using step 2 to build list of N-dimensional vectors
    4/ make a consineSim(query_vector, doc_vector) as a sort key for the result
    step 4 will use the formular I sent on Discord
    """


    ##### collect all docs containing more than 1 of the terms in query
    ##### buils matrix AKA list of vectors
    matrix = {} #{doc:{term:score}}
    query_terms=set(nltk.word_tokenize(user_input))
    for term in query_terms:
        for cur_pair in query(term,index):
            doc=cur_pair[0]
            score=cur_pair[1]
            if doc not in matrix:
                matrix[doc]={}
            for sub_term in query_terms:
                matrix[doc][sub_term]=0
                matrix[doc][term]=score

    ##### make a consineSim(...) as a sort key for the result
    query_vector={}
    query_tokens=nltk.word_tokenize(user_input)
    for term in query_tokens:
        term=lemmatize(term)
        if term in query_vector:
            query_vector[term] += 1
        else:
            query_vector[term] = 1

    def cosineSim(doc_vector): ##### length_normalize + dot_product = cosine_sim
        A,B,C=0,0,0
        for term in query_terms: #{doc:{term:score}} 
            A+=(doc_vector[term]*query_vector[term])
            B+=(query_vector[term]**2)
            C+=(doc_vector[term]**2)

        return A/(math.sqrt(B)*math.sqrt(C))

    """   
    query  | tf-idf  |  tf-idf
    ---------term1------term2-----
    doc1   | tf-idf  |  tf-idf
    doc2   | tf-idf  |  tf-idf
    """
    
    
    query_result=[]
    for doc in matrix:
        query_result.append(doc)
    query_result=sorted(query_result, key = lambda doc: -cosineSim(matrix[doc])) #add cosSim to key for sorting 


    #sort and return result
    result=""
    count=1
    global result_count
    result_count=len(query_result)
    with open("./WEBPAGES_RAW/bookkeeping.json","r", encoding='utf-8') as json_file:
        url_map=json.load(json_file)
    for doc in query_result:#--------> sort doc by TF-IDF
        result+=("\n{}: {}\n".format(doc,url_map[doc]))
        if count==20:
            break
        count+=1
    return result
def getNumDoc():
    global doc_count
    return doc_count
def getUniqueWordCount():
    return len(unique_words)
def getUrlCount():
    global result_count
    return result_count
def getDiskSize():
    return 0

if __name__ == "__main__":
    get_stopword()
    print("top 20 search result:\n"+displayQueryResult(input("please enter query: ")))
    print("number of documents searched:", getNumDoc())
    print("number of unique words processed:",getUniqueWordCount())
    print("number of result URLs:",getUrlCount())
    print("index's size on disk",getDiskSize())

# cd ../../winter2022/cs121/assignment3
# in computer computer science is not computer engineering