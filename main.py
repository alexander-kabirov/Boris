#!/usr/bin/env python
from Semantic_Parser_Executor.Execution.Executor import Executor
import pandas as pd


from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/',methods=['GET'])
def root():
    if 'text' in request.args:
        text = request.args['text']
    else:
        return "No text provided"
    if 'df' in request.args:
        df = request.args['df']
    else:
        return "No df provided"
    return str(executor.execute(text,df))


if __name__ == '__main__':
    #print(executor.execute('What is the largest legs with 1 catsize','zoo'))
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    # Flask's development server will automatically serve static files in
    # the "static" directory. See:
    # http://flask.pocoo.org/docs/1.0/quickstart/#static-files. Once deployed,
    # App Engine itself will serve those files as configured in app.yaml.
    zoo = pd.read_csv('dataframes/zoo2.csv')
    dataframes = {'zoo':zoo}
    executor = Executor(dataframes)
    print(executor.execute('What is the largest number of legs with 1 catsize and 0 toothed','zoo'))
    #print(executor.execute('What spiders are poisonous', 'zoo'))
    #app.run(host='0.0.0.0', port=8080) #debug=True
