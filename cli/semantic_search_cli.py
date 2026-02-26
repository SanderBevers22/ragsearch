#!/usr/bin/env python3

import argparse

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

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
