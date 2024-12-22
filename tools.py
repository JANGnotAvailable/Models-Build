import re
import nltk
import numpy as np
import cupy as cp

def softmax(inMatrix):
    m, n = cp.shape(inMatrix)
    outMatrix = cp.zeros((m, n))
    soft_sum = 0

    for idx in range(n):
        outMatrix[0, idx] = cp.exp(inMatrix[0, idx])
        soft_sum += outMatrix[0, idx]
    
    for idx in range(n):
        outMatrix[0, idx] = outMatrix[0, idx] / soft_sum


    return outMatrix



def text_clear(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9]", " ", text)
    text = re.sub(r" +", " ", text)
    text = text.strip()
    text = text.split(" ")
    text = [word for word in text if word not in stoplist]
    text = [PorterStemmer().stem(word) for word in text]
    text.append("eos")
    text = ["bos"] + text
    return text