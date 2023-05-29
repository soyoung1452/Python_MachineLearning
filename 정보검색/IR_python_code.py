#!/usr/bin/env python
# coding: utf-8

#### 구현한 함수 모듈 import

import pandas as pd
import IR_sy


#### stories.zip 문서 불러오기
# * 텍스트 문서 불러오기
# * 총 455개의 문서가 load

# 문서가 있는 폴더 경로
folder_path = 'stories'   
# 토큰화된 문서 dict와 파일명 dict 생성
tokenized_docs, filename_dic = IR_sy.read_documents(folder_path)    


# 파일 확장자별 문서 수 계산
extension_counts = {}
for document in filename_dic.values():
    extension = document.split('.')[-1]
    if extension in extension_counts:
        extension_counts[extension] += 1
    else:
        extension_counts[extension] = 1


# 파일 확장자 건수에 대한 통계표 생성
stats1 = pd.DataFrame({
    '파일 확장자': list(extension_counts.keys()),
    '문서 수': list(extension_counts.values())
})

stats1 = stats1.sort_values(by='문서 수', ascending=False)
stats1


#### SMIPI unigram inverted index 구현
inverted_index = IR_sy.spimi_invert(tokenized_docs)

# 단어별 문서 건수
stats2 = []
for term, doc_ids in inverted_index.items():
    doc_count = len(doc_ids)
    stats2.append((term, doc_count))
    
stats2.sort(key=lambda x: x[1], reverse=True)

for term, doc_count in stats2:
    print(f"Term: {term} | Document Count: {doc_count}")


#### boolean query 문서 검색
query ="Everyday Wish"

# AND query
and_result = IR_sy.boolean_and(inverted_index, query)

print("AND Boolean Results:", len(and_result), "건") 
for doc_id in and_result:
    print(f"Document {doc_id:3}: {filename_dic[doc_id]:15}")

# OR query
or_result = IR_sy.boolean_or(inverted_index, query)

print("OR Boolean Results:", len(or_result), "건") 
for doc_id in or_result:
    print(f"Document {doc_id:3}: {filename_dic[doc_id]:15}")

# AND NOT query
and_not_result = IR_sy.boolean_and_not(inverted_index, query)

print("AND NOT Boolean Results:", len(and_not_result), "건") 
for doc_id in and_not_result:
    print(f"Document {doc_id:3}: {filename_dic[doc_id]:15}")

# OR NOT query
or_not_result = IR_sy.boolean_or_not(tokenized_docs,inverted_index, query)

print("OR NOT Boolean Results:", len(or_not_result), "건") 
for doc_id in or_not_result:
    print(f"Document {doc_id:3}: {filename_dic[doc_id]:15}")


#### 문서의 TF-IDF 계산
tfidf_scores = IR_sy.calculate_tf_idf(tokenized_docs)
tfidf_scores


#### ranked 문서 검색
query

# Free text query : 'Everyday Wish'
ranked_result = IR_sy.ranked_retrieval(tfidf_scores, query)

print("Ranked Results:", len(ranked_result), "건") 
for doc_id, score in ranked_result:
    print(f"Document {doc_id:3}: {filename_dic[doc_id]:15}→   TF-IDF score = {score:.4f}")


#### 상위문서 TF-IDF 점수 확인
print(len(tfidf_scores[13]))
print(sorted(tfidf_scores[13].items(), key = lambda x:x[1], reverse=True)[:5])