import argparse
import json

from lib.hybrid_search import *
from lib.query_enhancement import *

from sentence_transformers import CrossEncoder

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser("summarize", help="Summarize results by use of LLM.")
    summarize_parser.add_argument("query",type=str,help="Search query for summarization")
    summarize_parser.add_argument("--limit",type=int,default=5,help="limit search")

    citation_parser = subparsers.add_parser("citations", help="Add citations")
    citation_parser.add_argument("query",type=str,help="Search query to add citations for")
    citation_parser.add_argument("--limit",type=int,default=5,help="limit search")

    question_parser = subparsers.add_parser("question",help="Ask a question and you will be answered")
    question_parser.add_argument("query",type=str,help="Question to ask")
    question_parser.add_argument("--limit",type=int,default=5,help="limit search")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            # do RAG stuff here
            with open("data/movies.json", "r") as f:
                documents = json.load(f)
            search = HybridSearch(documents["movies"])
            results = search.rrf_search(query,60,5)

            prompt = rag_results(query,results)
            response = enhance_query(prompt)
            
            retrieved_titles = [r["document"]["title"] for r in results]

            print("Search Results:")
            for t in retrieved_titles:
                print(f"  - {t}")

            print("\nRAG Response:")
            print(response)

        case "summarize":
            with open("data/movies.json", "r") as f:
                documents = json.load(f)
            search = HybridSearch(documents["movies"])
            results = search.rrf_search(args.query,60,args.limit)

            prompt = summarize_results(args.query,results)
            response = enhance_query(prompt)
            
            retrieved_titles = [r["document"]["title"] for r in results]

            print("Search Results:")
            for t in retrieved_titles:
                print(f"  - {t}")

            print("\nLLM Summary:")
            print(response)

        case "citations":
            with open("data/movies.json", "r") as f:
                documents = json.load(f)
            search = HybridSearch(documents["movies"])
            results = search.rrf_search(args.query,60,args.limit)

            prompt = citation_results(args.query,results)
            response = enhance_query(prompt)
            
            retrieved_titles = [r["document"]["title"] for r in results]

            print("Search Results:")
            for t in retrieved_titles:
                print(f"  - {t}")

            print("\nLLM Answer:")
            print(response)

        case "question":
            with open("data/movies.json", "r") as f:
                documents = json.load(f)
            search = HybridSearch(documents["movies"])
            results = search.rrf_search(args.query,60,args.limit)

            prompt = question_results(args.query,results)
            response = enhance_query(prompt)
            
            retrieved_titles = [r["document"]["title"] for r in results]

            print("Search Results:")
            for t in retrieved_titles:
                print(f"  - {t}")

            print("\nAnswer:")
            print(response)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
