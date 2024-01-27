import os
import subprocess
import lucene
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import json
import time
import org.apache.lucene.document as document
from datetime import timedelta
from java.io import *
from org.apache.lucene.search.similarities import BM25Similarity
from org.apache.lucene.analysis.tokenattributes import CharTermAttribute
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.analysis.en import EnglishAnalyzer
from org.apache.lucene.index import IndexWriter, IndexWriterConfig, IndexOptions
from org.apache.lucene.store import SimpleFSDirectory, FSDirectory
from collections import defaultdict


def end_execution(list_times,hashtags):
    for i in sorted(list_times.keys()):
        #print(f'index {i} documents takes {(list_times[i] / 60):.2f} minutes')
        print(f'index {i} documents takes {list_times[i]} seconds')
    print(f'\nTotal {len(hashtags.keys())} hashtags')
    end = time.time()
    print("time elapsed:",str(timedelta(seconds=end-start)))

def getElement(co):
    if len(co) > 0 and isinstance(co[0],list):
        return getElement(co[0])
    else:
        co[0] = float(co[0])
        co[1] = float(co[1])
        return co

def process_json_tokenize():
    data = []

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

def text_stemming(text):
    analyzer = EnglishAnalyzer()
    stream = analyzer.tokenStream("",text)
    stream.reset()
    tokens = []
    while stream.incrementToken():
        tokens.append(stream.getAttribute(CharTermAttribute.class_).toString())

    stream.close()
    return " ".join(tokens)

    
def write_text(data):
    with open("temp/tweet_text.txt","w") as file:
        for d in data:
            file.write(f'{d}^&&^||@@#$@@{data[d]}\n')
    file.close()

def document_insertion(tweets,id):
    doc = document.Document()

    text_data[id] = tweets["Text"]
    for key in tweets:
        if key == "Tweet_ID":
            doc.add(document.Field(key,id,metaType))
        elif key == "Text":
            doc.add(document.Field(key,tweets[key],text_field_type))
        elif key == "Coordinates":
            doc.add(document.LatLonPoint('Coordinates',tweets[key][1],tweets[key][0]))
            doc.add(document.Field('latitude',str(tweets[key][1]),document.TextField.TYPE_STORED))
            doc.add(document.Field('longtitude',str(tweets[key][0]),document.TextField.TYPE_STORED))
        elif key == "hashtags" or key =="Country" or key=="City":
            doc.add(document.Field(key,tweets[key],hashtag_field_type))
        elif key != "Text":
            doc.add(document.Field(key,tweets[key],document.TextField.TYPE_STORED))

    writer.addDocument(doc)
    return doc

if __name__ == "__main__":

    lucene.initVM()
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
    parser.add_argument('-r', required=True, help='rank algorithm')
    args = parser.parse_args()
    args.r = int(args.r)
    rank_option = {1:"index_tf_idf/",2: "index/"}

    df = process_json_tokenize()

    if os.path.exists(rank_option[args.r]):
        print('remove index folder\n')
        subprocess.run(f'rm -r {rank_option[args.r]}',shell=True)
    
    start = time.time()
    # create index object
    indexPath = File(rank_option[args.r]).toPath() # create index path
    indexDir = FSDirectory.open(indexPath) # create lucene store object to store index in hard disk
    writerConfig = IndexWriterConfig(StandardAnalyzer()) # create index configuration object. allow us to configure the index
    writerConfig.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
    if args.r == 2:
        writerConfig.setSimilarity(BM25Similarity())
    writer = IndexWriter(indexDir,writerConfig) # create index writer with the input of index path and configuration 

    id = 0
    text_data = {}

    metaType = document.FieldType()
    metaType.setStored(True)
    metaType.setTokenized(False)

    text_field_type = document.FieldType()
    text_field_type.setStored(False)
    text_field_type.setTokenized(True)
    text_field_type.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)

    hashtag_field_type = document.FieldType()
    hashtag_field_type.setStored(True)
    hashtag_field_type.setTokenized(True)
    hashtag_field_type.setIndexOptions(IndexOptions.DOCS_AND_FREQS)

    list_of_time = {}
    collect_hashtags = defaultdict(int)
    for tw in df:
        if id > 0 and id % 100000 == 0:
            end = time.time()
            seconds = timedelta(seconds=end-start)
            list_of_time[id] = seconds.total_seconds()
        tags = tw['hashtags'].split(" ")
        for t in tags:
            if t != "":
                collect_hashtags[t] += 1
        d = document_insertion(tw,str(id))
        id += 1

    writer.close()
    end_execution(list_of_time,collect_hashtags)
    write_text(text_data)