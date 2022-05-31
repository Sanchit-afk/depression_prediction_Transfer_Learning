# import libraries
from cs50 import SQL
from flask import *
from flask_mail import Mail, Message
from flask_session import Session
from flask_sslify import SSLify
from passlib.apps import custom_app_context as pwd_context
from helpers import *
import datetime
from flask import request
import random
import re
import os
import pickle
import texthero as hero
import pandas as pd
from sentence_transformers import SentenceTransformer

# configure application
app = Flask(__name__)
mail = Mail(app)

# redirect all http requests to https for secure connections
if 'DYNO' in os.environ:
    sslify = SSLify(app)

# set the secret key to use sessions
app.secret_key = os.urandom(24)

# ensure responses aren't cached
if app.config["DEBUG"]:
    @app.after_request
    def after_request(response):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Expires"] = 0
        response.headers["Pragma"] = "no-cache"
        return response


global count 
count = -1

global answers
answers=[]

#questions
def get_questions(count):
    questions = ["How are you feeling?", "During the past 4 weeks, what kind of problems did you face in your daily life? ","Generally people resort to drugs and alcohol to relieve stress. What are your views on this?",
    "What type of music genre do you like?", "How do you feel when a problem pops up unexpectedly?","What tense is your mind living in?",
    "Do you know what's behind your moods?","How's your focus?","How interested are you in your hobbies and activities? ","Have you thought about death or suicide?"]
    return questions[count]


def get_preds(answers):
    svm_classifier = pickle.load(open('depression_model.sav', 'rb'))
    df = pd.DataFrame(answers, columns = ['sample'])
    df['clean_data'] = hero.clean(df['sample'])
    sample_sent = df['clean_data'].tolist() 
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    samp_embeddings = model.encode(sample_sent)
    predictions = []
    for val in range(0,len(samp_embeddings)):
        predicted_value = svm_classifier.predict(samp_embeddings[val].reshape(1,-1))
        predictions.append(predicted_value)
    depress = 0
    normal = 0
    for val in range(0,len(samp_embeddings)):
        if predictions[val] == 0:
            normal+=1
        else:
            depress+=1
    return depress,normal 


# index route
@app.route("/")
def index():
    global count
    count=-1
    # return rendered index.html page with random books from Google Books API
    return render_template("index.html")


@app.route("/examp", methods=['GET','POST'])
def examp():
    # q1 = get_questions(count)
    global count
    if(count>-1):
        answer = request.form['textareas']
        print(answer)
        answers.append(answer)
    while(count<9):
        count+=1
        q2 = get_questions(count)

        # projectpath = request.form['projectFilepath']
        # your code
        
        return render_template("examp.html",question=q2)

    count=-1
    depress,normal = get_preds(answers)
    print(depress)
    print(normal)
    if depress>=5:
        predict = 'show signs'
    else:
        predict = 'do not show signs'
    # print(answers)
    # return rendered index.html page with random books from Google Books API
    return render_template("result.html",predict=predict)


# about route
@app.route("/about")
def about():

    # return rendered about.html page
    return render_template("about.html")


# help route
@app.route("/help")
def help():

    # return rendered blog.html page with all posts
    return render_template("services.html")


# contact route
@app.route("/contact")
def contact():

    # return rendered about.html page
    return render_template("contact.html")


# catch all other routes that doesn't exist
@app.errorhandler(404)
def page_not_found(e):

    # return rendered pageNotFound.html page
    return render_template("index.html")



if(__name__=="__main__"):
    app.run(debug=False)
    
