import lucene
import time
import re
import faiss
import json
from datetime import timedelta
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from java.io import *
from org.apache.lucene.search import IndexSearcher, BooleanQuery, BooleanClause
from org.apache.lucene.search.similarities import BM25Similarity
from org.apache.lucene.store import SimpleFSDirectory, FSDirectory
from org.apache.lucene.queryparser.classic import QueryParser, MultiFieldQueryParser
from org.apache.lucene.document import LatLonPoint
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.index import DirectoryReader
from transformers import AutoTokenizer, AutoModel
from flask import Flask, render_template, request
import torch


app = Flask(__name__,template_folder='views')

def end_execution(indx):
    end = time.time()
    print("time elapsed:",str(timedelta(seconds=end-start)))
    report[indx] = str(timedelta(seconds=end-start))
    

def extract_numbers(query):
    numbers = re.findall(r"[-]?\d*\.\d+|[-]?\d+",query)
    integers = []
    floats = []
    for i in numbers:
        if '.' in i:
            floats.append(i)
        else:
            integers.append(i)
    return floats, integers

def getElement(co):
    if len(co) > 0 and isinstance(co[0],list):
        return getElement(co[0])
    else:
        co[0] = float(co[0])
        co[1] = float(co[1])
        return co 
    
def getAllTweets():
    data = []
    # replace the filename
    with open("data/tweets.json","r") as file:
        tweets = json.load(file)
        for t in tweets:
            tw = {}
            for k,v in t.items():
                if k == 'Coordinates':
                    if v != 'null':
                        tw['Coordinates'] = getElement(v)
                elif k == 'Entities':
                    for ek, vk in v.items():
                        if ek == 'hashtags':
                            hashtags = []
                            for tag in vk:
                                hashtags.append(str(tag['text']))
                            tw['hashtags'] = " ".join(hashtags)
                        elif ek == 'media' or ek == 'urls':
                            if len(vk) > 0:
                                for key,value in vk[0].items():
                                    if key == "expanded_url":
                                        tw['url'] = str(value)
                elif v != 'null':
                    tw[k] = str(v)
            data.append(tw)
    return data

def convert_to_embedding(query):
    tokens = {'input_ids': [], 'attention_mask': []}
    new_tokens = tokenizer.encode_plus(query, max_length=512,
                                       truncation=True, padding='max_length',
                                       return_tensors='pt')
    tokens['input_ids'].append(new_tokens['input_ids'][0])
    tokens['attention_mask'].append(new_tokens['attention_mask'][0])
    tokens['input_ids'] = torch.stack(tokens['input_ids'])
    tokens['attention_mask'] = torch.stack(tokens['attention_mask'])
    with torch.no_grad():
        outputs = model(**tokens)
    x = (outputs.last_hidden_state * tokens['attention_mask'].unsqueeze(-1)).sum(1) / tokens['attention_mask'].sum(-1).unsqueeze(-1)
    #x = torch.mean(outputs.last_hidden_state, dim=1)
    
    return x[0] # assuming query is a single sentence

def read_text():
    text = {}
    with open("temp/tweet_text.txt","r") as file:
        start = False
        t = ""
        id = None
        first = True
        for line in file:
            element = line.strip().split("^&&^||@@#$@@")
            if len(element) == 2:
                start = False        
            if first:
                t += element[1]
                id = element[0]
                start = True
                first = False
            if not start and len(element) == 2:
                text[id] = t
                t = element[1]
                id = element[0]
                start = True
            elif start:
                t += line
        if id not in text:
            text[id] = t
    file.close()
    return text

def check_lat_lon(cor):
    if (cor[0] >= -90.0 and cor[0] <= 90.0) and (cor[1] >= -180.0 and cor[1] <= 180.0):
        return True
    return False

def create_query(query,fields):
    bQ = BooleanQuery.Builder()
    fields = ['Text','City','Country','hashtags','url']
    occurs = [BooleanClause.Occur.SHOULD for i in range(len(fields))]
    mutliField_parser = MultiFieldQueryParser(fields,analyzer)
    q1 = mutliField_parser.parse(query,fields,occurs,analyzer)
    bQ.add(q1,BooleanClause.Occur.SHOULD)
    
    numbers = re.findall(r"[-]?\d*\.\d+|[-]?\d+",query)
    if len(numbers) == 2:
        numbers[0] = float(numbers[0])
        numbers[1] = float(numbers[1])
        if check_lat_lon(numbers):
            q2 = LatLonPoint.newDistanceQuery("Coordinates", numbers[0], numbers[1], 100000.0)
            bQ.add(q2,BooleanClause.Occur.SHOULD)

    return bQ.build()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search',methods=['POST'])
def search():
    lucene.getVMEnv().attachCurrentThread()
    result = []
    input_query = request.form['query']
    method = request.form['search_engine']

    if method == 'bert':
        query_embedding = convert_to_embedding(input_query)

        D, I = faiss_index.search(query_embedding.reshape(1,model.config.hidden_size),k=5)
        for i in range(I.shape[1]):
            res_i = I[0,i]
            retrieved_tweet = tweets[res_i]
            result.append(retrieved_tweet)
            for field in retrieved_tweet:
                print(f'{field}: {retrieved_tweet[field]}')
    elif method == 'pylucene':
        input_query = re.sub(r'[^0-9a-zA-Z.-]+'," ",input_query)
        query = create_query(input_query,['Text','hashtags'])
        results = searcher.search(query,5)

        if results.totalHits.value < 5:
            query = create_query(input_query,['Text','City','Country','hashtags','url'])
            results = searcher.search(query,5)
        for hit in results.scoreDocs:
            d = reader.document(hit.doc)
            tweet = {}
            for f in d.getFields():
                name = f.name()
                if name == "hashtags" and f.stringValue() == "":
                    continue
                print(f'{name}: {f.stringValue()}')
                tweet[name] = f.stringValue()
                if name == "Tweet_ID":
                    tweet['Text'] = original_text[f.stringValue()]
            result.append(tweet)

    return render_template('search_results.html',query=input_query, results=result, method=method)


if __name__ == "__main__":
    lucene.initVM()
    print('Loading tweet data and faiss index')
    original_text = read_text()
    tweets = getAllTweets()
    faiss_index = faiss.read_index("index_bert/embed_docs_roberta_cosim.index")
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-distilroberta-v1') # you can change the model here
    model = AutoModel.from_pretrained('sentence-transformers/all-distilroberta-v1')

    indexPath = File("index_tf_idf/").toPath()
    indexDir = FSDirectory.open(indexPath)

    # search the index
    analyzer = StandardAnalyzer()
    reader = DirectoryReader.open(indexDir)
    searcher = IndexSearcher(reader)
    
    # localhost:8080 or 127.0.0.1:8080
    #app.run(port=8080) 

    #try for new ip&port
    app.run(host='0.0.0.0', port=8888)

