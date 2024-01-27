import lucene
import time
import re
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
from collections import defaultdict


def end_execution():
    end = time.time()
    print("time elapsed:",str(timedelta(seconds=end-start)))

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
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
    parser.add_argument('-r', required=True, help='rank algorithm')
    args = parser.parse_args()
    args.r = int(args.r)
    rank_option = {1:"index_tf_idf/",2: "index/"}

    original_text = read_text()

    indexPath = File(rank_option[args.r]).toPath()
    indexDir = FSDirectory.open(indexPath)

    # search the index
    analyzer = StandardAnalyzer()
    reader = DirectoryReader.open(indexDir)
    searcher = IndexSearcher(reader)

    # set different ranking algorithm. Default is tf-idf
    if args.r == 2:
        searcher.setSimilarity(BM25Similarity())

        
    while True:
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
        for hit in results.scoreDocs:
            d = reader.document(hit.doc)
            print('---------------------------')
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
            print('---------------------------')

        end_execution()
        print("-------------end-------------")
