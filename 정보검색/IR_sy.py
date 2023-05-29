import os
import glob
import re
import math
import string
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from collections import Counter, defaultdict
from tqdm import tqdm
import time


def preprocess_text(txt):
    txt = re.sub(f"[{re.escape(string.punctuation)}]", "", txt)        # 문장 부호 제거
    txt = re.sub(r"\d+", "", txt)                                      # 숫자 제거
    txt = txt.replace("\n", " ")                                       # 개행 문자 공배 대체
    txt = txt.lower()                                                  # 소문자 변환
    tokens = word_tokenize(txt)                                        # 토큰화하여 단어로 분리
    stop_words = set(stopwords.words("english"))                       
    tokens = [token for token in tokens if token not in stop_words]  # 불용어 제거
    lemmatizer = WordNetLemmatizer()  
    tokens = [lemmatizer.lemmatize(token) for token in tokens]        # 표제어 변환
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]                # 어간 추출
    processed_text = tokens                                           # 전처리 완료
    
    return processed_text


def read_documents(folder_path):
    tokenized_docs = dict()  # 토큰화 문서 dict
    filename_dic = dict()    # 문서 파일명 dict
    doc_id = 1               # 문서ID 초깃값
    
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='cp949', errors='replace') as file:
            data = file.read()
            pre_data = preprocess_text(data)       # 전처리 수행
            tokenized_docs[doc_id] = pre_data      # 토큰화 문서 dict
            filename_dic[doc_id] = filename        # 문서 파일명 dict
        doc_id += 1
 
    return tokenized_docs, filename_dic


def spimi_invert(tokenized_docs):
    inverted_index = {}     # inverted index dict
    
    for doc_id, doc in tqdm(tokenized_docs.items()):
        for term in doc:
            if term not in inverted_index:
                inverted_index[term] = []      # 현재 단어가 dict에 없는 경우, 빈 리스트 값 추가
            if doc_id not in inverted_index[term]:
                inverted_index[term].append(doc_id)  # 문서 ID dict에 추가
        
    return inverted_index


def boolean_and(index, query):
    query_tokens = query.lower().split()
    result = None
    for term in query_tokens:
        if term in index:
            if result is None:
                result = set(index[term])
            else:
                result.intersection_update(index[term])
                # intersection_update
                # 집합(set) 객체의 메서드 중 하나
                # 집합 객체 자체를 다른 집합과의 교집합으로 업데이트
        
    return result
        
def boolean_or(index, query):
    query_tokens = query.lower().split()
    result = []
    
    for term in query_tokens:
        if term in index:
            result += index[term] # term의 인덱스 모두 result 넣기
    result = list(set(result))   # 중복 값 없애기
    return result
        
        
def boolean_and_not(index, query):
    # 질의가 두 개의 용어로 이루어져 있지 않으면 빈 리스트 반환
    # 유효한 질의 형태 아닌 경우 처리하기 위함
    query_tokens = query.lower().split()
    if len(query_tokens)!= 2: return []
    result = []
    token1, token2 = query_tokens
    
    # term1이 존재하고 term2가 존재하지 않는 문서ID들을 반환
    if token1 in index and token2 in index:
        result = list(set(index[token1]) - set(index[token2]))
    # term1만 index에 존재하는 경우, term1이 존재하는 ID 반환
    elif token1 in index:
        result = index[token1]
    # term1과 term2 모두 index에 존재하지 않는 경우 빈리스트 반환
    else:
        result = []
    
    return result
        
        
def boolean_or_not(tokenized_docs, index, query):
    query_tokens = query.lower().split()
    if len(query_tokens)!= 2: return []
    result = []
    token1, token2 = query_tokens

    if token1 in index and token2 in index:
        result = list(set(index[token1] + list(set(range(1, len(tokenized_docs)+1)) - set(index[token2]))))
    elif token1 in index:
        result= index[token1]
    elif token2 in index:
        result = list(set(range(1, len(tokenized_docs)+1)) - set(index[token2]))
    else:
        result = []
    
    return result
        

def calculate_tf_idf(tokenized_docs):
    # 문서 내 단어 빈도 계산
    doc_word_counts = [Counter(doc) for doc in tokenized_docs.values()]
    # 문서 내 전체 단어 수 계산
    doc_total_words = [len(doc) for doc in tokenized_docs.values()]
    # 문서 집합 크기 계산
    num_documents = len(tokenized_docs)
    
    # TF 계산 (Term Frequency)
    # 특정 단어가 문서 내에서 얼마나 자주 등장하는지
    # 해당 단어 등장 회수/전체 단어수
    tf_scores = []
    for doc_word_count, total_words in zip(doc_word_counts, doc_total_words):
        tf_scores.append({word: count / total_words for word, count in doc_word_count.items()})
    
    # IDF 계산 (Inverse Document Frequency)
    # 특정 단어가 전체 문서 집합에서 얼마나 희귀한지
    # 로그 역수로 계산되며, 전체 문서 집합에서 해당 단어가 등장하는 문서의 역수를 취한 값
    # 희귀한 경우에 더 큰 값이 된다.
    idf_scores = {}
    for doc_word_count in doc_word_counts:
        for word in doc_word_count:
            if word not in idf_scores:
                idf_scores[word] = math.log(num_documents / sum(1 for doc_count in doc_word_counts if word in doc_count))
    
    # TF-IDF 계산
    # 단어의 중요도를 나타내는 지표
    # 특정 단어가 문서 내에서 얼마나 자주 등장하는지를 고려하면서 동시에 해당 단어가 전체 문서 집합에서 얼마나 희귀한지를 고려한다.
    # 특정 문서 내에서 자주 등장하는 단어이면서 전체 문서 집합에서 희귀한 단어 일수록 해당 단어의 TF-IDF 값은 상대적으로 크게 된다.
    tfidf_scores = {}
    for doc_id, tf_score in tqdm(enumerate(tf_scores)):
        tfidf_scores[doc_id+1] = {word: tf_score[word] * idf_scores[word] for word in tf_score if word in idf_scores}
    
    return tfidf_scores

  
def ranked_retrieval(tfidf_scores, query):
    # 검색어 토큰
    query_tokens = query.lower().split()
    # 문서 검색어 스코어 계산
    document_scores = {}
    # 각 문서에 대해 검색어에 포함된 각 토큰의 tf-idf 값을 합산
    for doc_id, doc_tfidf in tfidf_scores.items():
        score = sum(doc_tfidf[token] for token in query_tokens if token in doc_tfidf)
        document_scores[doc_id] = score
    # 검색 결과 랭킹
    ranked_result = sorted(document_scores.items(), key = lambda x:x[1], reverse=True)
    return ranked_result