#!/usr/bin/env python3

import argparse
import json
import math
import os
import string

from inverted_index import InvertedIndex
from preprocessing import preprocess

def matching_logic(query,data,stopwords):
    query_tokens = query.split()
    title_tokens = data.split()

    query_tokens = [t for t in query_tokens if t not in stopwords and t != ""]
    title_tokens = [t for t in title_tokens if t not in stopwords and t != ""]

    stemmer = PorterStemmer()

    query_tokens = [stemmer.stem(t) for t in query_tokens]
    title_tokens = [stemmer.stem(t) for t in title_tokens]

    for token in query_tokens:
        for word in title_tokens:
            if token in word:
                return True
    return False

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    build_parser = subparsers.add_parser("build", help="Build inverted index")
    tf_parser = subparsers.add_parser("tf", help="Get frequency of token in document")
    tf_parser.add_argument("doc_id",type=int,help="Document ID")
    tf_parser.add_argument("term", type=str,help="Term to check frequency for")
    idf_parser = subparsers.add_parser("idf", help="Inverse document frequency command")
    idf_parser.add_argument("term",type=str,help="Term to check frequency in all documents")
    tfidf_parser = subparsers.add_parser("tfidf",help="Overall scoring for term in document")
    tfidf_parser.add_argument("doc_id",type=int,help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to check")

    args = parser.parse_args()
    path = os.path.join(os.path.dirname(__file__),"..","data","movies.json")
    stopwords_path = os.path.join(os.path.dirname(__file__),"..","data","stopwords.txt")
    with open(path) as f:
        data = json.load(f)

    with open(stopwords_path) as f:
        stopwords = f.read().splitlines()
        
    match args.command:
        case "search":
            print(f"Searching for: {args.query}")

            index = InvertedIndex()

            try:
                index.load()
            except FileNotFoundError:
                print("index not found. run build first.")
                return

            tokens = preprocess(args.query,stopwords)

            results = []
            seen = set()

            for token in tokens:
                doc_ids = index.get_documents(token)

                for doc_id in doc_ids:
                    if doc_id not in seen:
                        results.append(index.docmap[doc_id])
                        seen.add(doc_id)

                    if len(results) == 5:
                        break
                if len(results) == 5:
                    break

            for i,movie in enumerate(results,start=1):
                print(f"{i}. {movie['title']} (ID: {movie['id']})")
        
        case "build":
            index = InvertedIndex()
            index.build(data["movies"],stopwords)
            index.save()
            print("index built")

        case "tf":
            index = InvertedIndex()
            try:
                index.load()
            except FileNotFoundError:
                raise FileNotFoundError("Error loading index")
            
            tf = index.get_tf(args.doc_id,args.term)
            print(tf)

        case "idf":
            index = InvertedIndex()
            try:
                index.load()
            except FileNotFoundError:
                raise FileNotFoundError("Error loading index")
            
            tokens = preprocess(args.term)
            if len(tokens) != 1:
                raise ValueError("Expected single token")
            token = tokens[0]

            totaldocs = len(index.docmap)
            totalmatch = len(index.get_documents(token))

            idf = math.log((totaldocs+1)/(totalmatch+1))
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        case "tfidf":
            index = InvertedIndex()
            try:
                index.load()
            except FileNotFoundError:
                raise FileNotFoundError("Error loading index")
            
            tokens = preprocess(args.term)
            if len(tokens) != 1:
                raise ValueError("Expected single token")
            token = tokens[0]

            totaldocs = len(index.docmap)
            totalmatch = len(index.get_documents(token))

            idf = math.log((totaldocs+1)/(totalmatch+1))
            tf = index.term_frequencies[args.doc_id][args.term]
            tfidf = tf * idf

            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tfidf:.2f}")


        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
