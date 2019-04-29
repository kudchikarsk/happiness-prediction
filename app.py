from flask import Flask
app = Flask(__name__)
from flask import render_template
from form import ReusableForm
from keras.preprocessing import sequence
from keras.models import load_model
import nltk
import pickle
import numpy as np

app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'
word2index=pickle.load( open( 'word2index.p', "rb" ) )
index2word=pickle.load( open( 'index2word.p', "rb" ) )
model_filename="model.h5"
MAX_SENTENCE_LENGTH=500


# Home page
@app.route("/", methods=['GET', 'POST'])
def home():
    """Home page of app with form"""
    # Create form
    form = ReusableForm()
    isHappy = True

    if form.is_submitted():
        #form.text.data
        isHappy = predict(form.text.data)

    # Send template information to index.html
    return render_template('index.html', form=form, isHappy=isHappy)

def predict(text):
    model=load_model(model_filename)
    #model.compile(loss="binary_crossentropy",optimizer="rmsprop",metrics=["accuracy"])
    ntext = normalize(np.array([text]))
    predictions=model.predict(ntext)
    predictions=denormalize_response(predictions)
    return denormalize_response(predictions)[0]

def denormalize_response(predictions):
    return [True if x > 0.5  else False for x in predictions]

def normalize(train_description):
    X=np.empty((train_description.size,),dtype=list)
    i=0
    for sentence in train_description:
        words=nltk.word_tokenize(sentence.lower())
        seqs=[]
        for word in words:
            if word in word2index:
                seqs.append(word2index[word])
            else:
                seqs.append(word2index["UNK"])
        X[i]=seqs
        i+=1
    return sequence.pad_sequences(X,maxlen=MAX_SENTENCE_LENGTH)
	
app.run(host='0.0.0.0', port=50000, debug=True)
