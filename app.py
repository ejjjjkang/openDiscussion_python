
#load_libraries
import os
from konlpy.tag import Okt
from flask import Flask, request, jsonify
from flask_cors import CORS

import re
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.python.keras.backend as K



stop_words = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']  # 불용어 지정

#Instantiate_Flask
app = Flask(__name__)
CORS(app)


def data_preprocessing(text):
    test_text = []

    # loading tokenizer
    with open('tokenizer_0505.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    text = re.sub('ㅋ', '', text)
    text = re.sub('!', '', text)
    text = re.sub(',', '', text)
    text = text.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")

    okt = Okt()

    temp_text = okt.morphs(text, stem= True) #stem이 일정 수준의 정규화를 수행시켜준다

    #불용어 제거
    temp_text = [ word for word in temp_text if not word in stop_words]
    test_text.append(temp_text)

    #정수 인코딩
    container = test_text.copy()


    container = tokenizer.texts_to_sequences(container)
    container = pad_sequences(container)

    print('text:', container)

    return container


def predict_model(text):


    token_word = data_preprocessing(text)

    print("====tokenize_complete=====")

    loaded_model = tf.keras.models.load_model("./best_model_0505.h5")
    print('Weights loaded...')
    loaded_model.summary()

    #graph = tf.get_default_graph() # If this step is omitted, an exception may occur during the predict step.

    scores = loaded_model.predict(token_word)
    scores = round(float(scores),2)
    if(scores < 0.5):
        return ["negative", scores]
    else:
        return ["positive", scores]


@app.route('/process', methods=["GET", "POST"])

def process():
    content = request.json

    if content['words'] is not None:
        result = predict_model(content['words'])
        return_result = {'Result' : result}
        K.clear_session()
    return jsonify(return_result)

if __name__ == '__main__':
    app.run('0.0.0.0', port=5000)
