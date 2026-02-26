#!/usr/bin/env python3

import argparse
import re

from lib.semantic_search import *

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    verify_parser = subparsers.add_parser("verify",help="verify model")
    embed_parser = subparsers.add_parser("embed_text",help="text embedder")
    embed_parser.add_argument("text",type=str,help="text to embed")
    verify_embeddings_parser = subparsers.add_parser("verify_embeddings",help="verify embeddings")
    query_embed_parser = subparsers.add_parser("embedquery", help="query embedder")
    query_embed_parser.add_argument("query",type=str,help="query to embed")
    search_parser = subparsers.add_parser("search",help="Search documents on query")
    search_parser.add_argument("query",type=str,help="query to search")
    search_parser.add_argument("--limit", type=int,default=5,help="optional limit character")
    chunk_parser = subparsers.add_parser("chunk",help="chunk a text")
    chunk_parser.add_argument("text",type=str,help="text to chunk")
    chunk_parser.add_argument("--chunk-size",type=int,default=200,help="chunk size for text")
    chunk_parser.add_argument("--overlap",type=int,default=5,help="Overlap size of the created chunks, to not lose context.")
    semantic_chunk_parser = subparsers.add_parser("semantic_chunk",help="Semantic chunk a text")
    semantic_chunk_parser.add_argument("text",type=str,help="text to chunk")
    semantic_chunk_parser.add_argument("--max-chunk-size",type=int,default=4,help="max chunk size")
    semantic_chunk_parser.add_argument("--overlap",type=int,default=0,help="overlap of chunks")
    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()

        case "embed_text":
            embed_text(args.text)

        case "verify_embeddings":
            verify_embeddings()

        case "embedquery":
            embed_query_text(args.query)

        case "search":
            searcher = SemanticSearch()
            with open("data/movies.json", "r") as f:
                documents = json.load(f)
            searcher.load_or_create_embeddings(documents["movies"])
            results = searcher.search(args.query,args.limit)
            for i,result in enumerate(results,start=1):
                print(f"{i}. {result['title']} (score: {result['score']:.4f})")
                print(f"   {result['description']}\n")

        case "chunk":
            words = args.text.split()
            chunks=[]

            if args.overlap <= 0:
                overlap = 0
            else:
                overlap = args.overlap
            i=0
            while i < len(words):
                if i < args.chunk_size:
                    chunk = " ".join(words[i:i+args.chunk_size])
                    chunks.append(chunk)
                    i+=args.chunk_size
                else:
                    i-=overlap
                    chunk = " ".join(words[i:i+args.chunk_size])
                    chunks.append(chunk)
                    i+=args.chunk_size

            print(f"Chunking {len(args.text)} characters")
            for idx,chunk in enumerate(chunks,start=1):
                print(f"{idx}. {chunk}")

        case "semantic_chunk":
            lines = re.split(r"(?<=[.!?])\s+",args.text)
            chunks=[]

            if args.overlap > 0:
                overlap = args.overlap
            else:
                overlap = 0

            i=0
            while i < len(lines):
                if i < args.max_chunk_size:
                    chunk = " ".join(lines[i:i+args.max_chunk_size])
                    chunks.append(chunk)
                    i+=args.max_chunk_size
                else:
                    i-=overlap
                    chunk = " ".join(lines[i:i+args.max_chunk_size])
                    chunks.append(chunk)
                    i+=args.max_chunk_size
            print(f"Semantically chunking {len(args.text)} characters.")
            for idx,chunk in enumerate(chunks,start=1):
                print(f"{idx}. {chunk}")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
