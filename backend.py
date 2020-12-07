from flask import Flask, render_template, request, redirect, url_for , Response

from backend import db
db.create_all()
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB


app = Flask(__name__)


@app.route("/",  methods=['GET'])
def index():
    return render_template("index.html");


@app.route("/Predict" ,  methods=['POST'])
def Predict():
    user_input = request.form['text']
    df = pd.read_csv('https://raw.githubusercontent.com/earth4950/datasetbig/main/train.csv')
    conversion_dict = {0: 'HQ', 1: 'LQ_EDIT', 2: 'LQ_CLOSE'}
    df['Body'] = df['Y'].replace(conversion_dict)
    # print(df.label.value_counts())

    # Train test split
    x_train, x_test, y_train, y_test = train_test_split(
    df['Body'], df['Y'], test_size=0.25, random_state=7, shuffle=True)
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.75)

    # แปลงเป็นตัวเลข
    vec_train = tfidf_vectorizer.fit_transform(x_train.values.astype('U'))
    vec_test = tfidf_vectorizer.transform(x_test.values.astype('U'))

    # Train Model
    pac = PassiveAggressiveClassifier(max_iter=50)
    pac.fit(vec_train, y_train)
    model = MultinomialNB()
    model.fit(vec_train, y_train)

    # Predict
    user_input_tranform = tfidf_vectorizer.transform([user_input])
    y_predict = pac.predict(user_input_tranform)

    return render_template("predict.html" , text = user_input , predict = y_predict );



# Run Server
if __name__ == "__main__":
    app.run(debug=True)
