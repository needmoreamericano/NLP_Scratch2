# coding: utf-8
import sys
sys.path.append('..')
import numpy as np
import common
import os
#from common.np import *


def preprocess(text):
    text = text.lower()
    text = text.replace('.', ' .')
    words = text.split(' ')

    word_to_id = {}
    id_to_word = {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus = np.array([word_to_id[w] for w in words])

    return corpus, word_to_id, id_to_word

# 동시발생 행렬 만드는 함수

def create_co_matrix(corpus, vocab_size, window_size = 1) :
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size))

    for idx, word_id in enumerate(corpus) :
        for i in range(1, window_size+1) :
            left_idx = idx - 1
            right_idx = idx + 1

            if left_idx >= 0 :
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1

            if right_idx < corpus_size :
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1
    return co_matrix


# 코사인 유사도

def cos_similarity(x, y, eps = 1e-8) :
    nx = x / (np.sqrt(np.sum(x**2)) + eps)
    ny = y / (np.sqrt(np.sum(y**2)) + eps)

    return np.dot(nx, ny)

# 코사인 유사도 기반 유사도 랭킹

def most_similar(query, word_to_id, id_to_word, word_matrix, top = 5) :
    if query not in word_to_id :
        print('%s 를 찾을 수 없습니다.' %query)
        return

    print('\n[query]' + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    ## 유사도 계산
    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size) :
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    ## 내림차순 출력
    count = 0
    for i in (-1*similarity).argsort() :
        if id_to_word[i] == query :
            continue
        print('%s : %s' %(id_to_word[i], similarity[i]))

        count += 1
        if count >= top :
            return

# PPMI : PPMI(x, y) = max(0, PMI(x,y))

## C = 동시발생 행렬
## verbose = 진행상황 출력 여부
def ppmi(C, verbose = False, eps = 1e-8) :
    M = np.zeros_like(C)
    N = np.sum(C)
    S = np.sum(C, axis = 0)
    total = C.shape[0] * C.shape[1]
    cnt = 0

    for i in range(C.shape[0]) :
        for j in range(C.shape[1]) :
            pmi = np.log2(C[i,j]*N / (S[j]*S[i] + eps))
            M[i, j] = max(0, pmi)

            if verbose :
                cnt += 1
                if cnt % (total//100) == 0 :
                    print('%.1f%% 완료' %(100*cnt/total))

    return M

# corpus로 부터 맥락과 타깃 설정
 
def create_contexts_target(corpus, window_size = 1) :
    target = corpus[window_size : -window_size]
    contexts = []
    
    for idx in range(window_size, len(corpus) - window_size) :
        cs = []
        for t in range(-window_size, window_size + 1) :
            if t == 0 :
                continue
            cs.append(corpus[idx + t])
        contexts.append(cs)
        
    return np.array(contexts), np.array(target)


def convert_one_hot(corpus, vocab_size):
    '''원핫 표현으로 변환
    :param corpus: 단어 ID 목록(1차원 또는 2차원 넘파이 배열)
    :param vocab_size: 어휘 수
    :return: 원핫 표현(2차원 또는 3차원 넘파이 배열)
    '''
    N = corpus.shape[0]

    if corpus.ndim == 1:
        one_hot = np.zeros((N, vocab_size), dtype=np.int32)
        for idx, word_id in enumerate(corpus):
            one_hot[idx, word_id] = 1

    elif corpus.ndim == 2:
        C = corpus.shape[1]
        one_hot = np.zeros((N, C, vocab_size), dtype=np.int32)
        for idx_0, word_ids in enumerate(corpus):
            for idx_1, word_id in enumerate(word_ids):
                one_hot[idx_0, idx_1, word_id] = 1

    return one_hot