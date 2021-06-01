import joblib
import numpy as np
import pandas as pd
from flask import Flask,request,render_template
from pandas.core.frame import DataFrame

app = Flask(__name__)
tfidf_vec = joblib.load('tfidf_vec.pkl')
kdtree_model = joblib.load('kdtree_model.pkl')
data_info = pd.read_csv(r'C:/Users/Darshana/Desktop/DSC_WKND20092020/NLP/infromation retrival/famous_people.csv')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    output = ' '
    int_text = [x for x in request.form.values()]
    
    distance,idx = kdtree_model.query(tfidf_vec.transform(int_text).toarray(),k=5)

    # for i,val in list(enumerate(idx[0])):
        # print(f'Name:{data_info["Name"][val]}')
        # print(f'Distance : {distance[0][i]}')
        # print(f'URI : {data_info["URI"][val]}')
         
    # return render_template('index.html',prediction_text='The Documents are :{}'.format(output))
    return render_template('index.html',idx = list(enumerate(idx[0])),DataFrame=data_info)

if __name__ == "__main__":
    app.run(debug=True)