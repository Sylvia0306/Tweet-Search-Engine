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
import torch


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

if __name__ == "__main__":
    lucene.initVM()
    # parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
    # parser.add_argument('-i', required=True, help='index')
    # parser.add_argument('-r', required=True, help='ranking algorithm')
    # args = parser.parse_args()
    # args.i = int(args.i)
    # args.r = int(args.r)
    
    rank_option = {1:"index_tf_idf/",2: "index/"}
    index_option = {1: 'Lucence', 2:'Faiss', 3: 'report'}
    print('Loading tweet data and faiss index')
    original_text = read_text()
    tweets = getAllTweets()
    faiss_index = faiss.read_index("index_bert/embed_docs_roberta_cosim.index")

    report = {'Lucence': 0, 'Faiss':0}

    while True:
        try:
            selected_index = input('Please select the index you want to use: 1. Lucence 2. Faiss\n')
            if index_option[int(selected_index)] == 'Lucence':
                print("You select Lucence Index")
                selected_rank = input('Please select the score function you want to use: 1. tf-idf 2. BM-25\n')
                
                indexPath = File(rank_option[int(selected_rank)]).toPath()
                indexDir = FSDirectory.open(indexPath)

                # search the index
                analyzer = StandardAnalyzer()
                reader = DirectoryReader.open(indexDir)
                searcher = IndexSearcher(reader)

                if int(selected_rank) == 2:
                    searcher.setSimilarity(BM25Similarity())


                input_query = input("Please Enter a query:\n")
                # replace all non-alphanumeric characters
                start = time.time()
                input_query = re.sub(r'[^0-9a-zA-Z.-]+'," ",input_query)
                if input_query == "":
                    continue
                # add wildcards to each word
                #input_query = " ".join(word+"*" if not word.isdigit() and not word[-1].isdigit() else word for word in input_query.split(" "))
                
                #query = QueryParser('Text',analyzer).parse(input_query)
                query = create_query(input_query,['Text','hashtags'])
                results = searcher.search(query,5)
                # In case there is no match in Text field
                if results.totalHits.value < 5:
                    print(results.totalHits.value)
                    query = create_query(input_query,['Text','City','Country','hashtags','url'])
                    results = searcher.search(query,5)

                #print(results.totalHits.value)
                #print(f'Query: {query}\n')
                print('================Top 5 Results================')
                for hit in results.scoreDocs:
                    d = reader.document(hit.doc)
                    print(f'Document {hit.doc} - Score: {hit.score}')
                    tid = ""
                    for f in d.getFields():
                        name = f.name()
                        if name == "hashtags" and f.stringValue() == "":
                            continue
                        print(f'{name}: {f.stringValue()}')
                        if name == "Tweet_ID":
                            tid = f.stringValue()
                    
                    print(f'Text: {original_text[tid]}')
                    print('------------------------------------------------------')

                end_execution(index_option[int(selected_index)])
                print()
            

            
            elif index_option[int(selected_index)] == 'Faiss':
                # if you change the pretrained model, you should also modify this
                tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-distilroberta-v1') # you can change the model here
                model = AutoModel.from_pretrained('sentence-transformers/all-distilroberta-v1')

                
                input_query = input("Please Enter a query:\n")
                start = time.time()
                query_embedding = convert_to_embedding(input_query)

                D, I = faiss_index.search(query_embedding.reshape(1,model.config.hidden_size),k=5)
            
                print('================Top 5 Results================')
                for i in range(I.shape[1]):
                    res_i = I[0,i]
                    text = tweets[res_i]
                    print(f'Cosine Similarity Score: {D[0,i]:.2f}')
                    for field in text:
                        print(f'{field}: {text[field]}')
                    
                    print('------------------------------------------------------')

                end_execution(index_option[int(selected_index)])
                print()
            else:
                for key, value in report.items():
                    print(f'{key} spent {value} on the last query search')
                break
        except Exception as e:
            print(e)


