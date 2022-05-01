import re
import json
import nltk
import pickle
import random
import warnings
import numpy as np
from tashaphyne import normalize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
warnings.filterwarnings('ignore')
from flask import Flask, render_template
from flask_socketio import SocketIO,send, emit
from tensorflow.keras.models import load_model
#from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load pkl model
detect_model = pickle.load(open('../Detect Dialect/Models/log_model.pkl', 'rb'))
dict = { 0:'Egyption', 1:'Maghreb', 2:'Gulf', 3:'Levantine', 4:'Others'}


app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)


@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('send_message')
def handle_message(data):

    # print('received message: ' + str(data))
    text = str(data)
    
    #preprocessing text
    pre_text = preprocessing_text(text)

    # making prediction for log model
    prediction = detect_model.predict(get_tfidf(pre_text))
    log_output = dict[prediction[0]]
    
    reply = ""
    if log_output == "Egyption":
        reply = end_response(data, '../Conversation/data/EG_data.json', '../Conversation/words/eg_words.pkl', '../Conversation/classes/eg_classes.pkl', '../Conversation/models/eg_model.h5')
    
    elif log_output == "Gulf":
        reply = end_response(data, '../Conversation/data/GU_data.json', '../Conversation/words/gu_words.pkl', '../Conversation/classes/gu_classes.pkl', '../Conversation/models/gu_model.h5')
    
    elif log_output == "Maghreb":
        reply = end_response(data, '../Conversation/data/MG_data.json', '../Conversation/words/mg_words.pkl', '../Conversation/classes/mg_classes.pkl', '../Conversation/models/mg_model.h5')
    
    elif log_output == "Levantine":
        reply = end_response(data, '../Conversation/data/LE_data.json', '../Conversation/words/le_words.pkl', '../Conversation/classes/le_classes.pkl', '../Conversation/models/le_model.h5')
    
    else:
        reply = end_response(data, '../Conversation/data/Arabic_data.json', '../Conversation/words/th_words.pkl', '../Conversation/classes/th_classes.pkl', '../Conversation/models/th_model.h5')
    

    socketio.emit('recive_message', reply)


## For detect dialect
def cleaning(text): 
    newtext = re.sub('([@A-Za-z0-9_])|[^\w\s]|#|http\S+|', '', text).replace('\n',' ').lstrip().rstrip()
    return re.sub(r'(.)\1+', r'\1', newtext)

def preprocessing_text(text):
    text = normalize.normalize_searchtext(text)
    return cleaning(text)

def get_tfidf(text):
    tfidf = pickle.load(open('../Detect Dialect/Models/tfidf.pkl', 'rb'))
    text_tfidf = tfidf.transform([text])
    return text_tfidf



## For converstation 
def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))
  
def predict_class(sentence, model, words, classes):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break

    return result

def chatbot_response(msg, conv_model, intents, words, classes):
    ints = predict_class(msg, conv_model, words, classes)
    res = getResponse(ints, intents)
    return res

def end_response(msg, path_of_intents, path_of_words, path_of_classes, path_of_model):
    intents = json.loads(open(path_of_intents, encoding='utf-8').read())
    words = pickle.load(open(path_of_words,'rb'))
    classes = pickle.load(open(path_of_classes,'rb'))
    model = load_model(path_of_model)

    res = chatbot_response(msg, model, intents, words, classes)

    return res

if __name__ == '__main__':
    socketio.run(app)  