from flask import request, jsonify, render_template
from application import app
from nb_spam_classifier import NaiveBayesSpamFilter 
import pandas as pd
import numpy as np

model = NaiveBayesSpamFilter()
model.train()

df = pd.read_csv('spam_or_not_spam.csv')

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/classify_text', methods=['GET','POST'])
def classify_text():
    global df
    if request.method == 'POST':
        if request.form["action"] == "Report spam":
            text = df.tail(1)['email'].values[0]
            df.loc[np.where(df['email']==text)[0],'label']='SPAM'
            model.calculate_word_frequencies(text, 'SPAM')
            return render_template('report_spam.html')

        text = request.form['Text']
        result = model.classify(text)
        df = df.append(pd.DataFrame({'email': [f'{text}'], 'label': [f'{result}']}), ignore_index=True)

        if result == 'SPAM':
            return render_template('spam.html')
        else:
            return render_template('not_spam.html', text=f'{text[:30]}...')

    return render_template('index.html')
