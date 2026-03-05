import argparse
import json

from lib.hybrid_search import *
from lib.query_enhancement import *

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    normalize = subparsers.add_parser("normalize",help="verify model")
    normalize.add_argument("score",type=float,nargs="+",help="Normalize scores")
    weighted = subparsers.add_parser("weighted-search",help="Hybrid search with weights")
    weighted.add_argument("query",type=str,help="Query to search for")
    weighted.add_argument("--alpha",type=float,default=0.5,help="Weight of semantic vs keyword")
    weighted.add_argument("--limit",type=int,default=5,help="limit search results")
    rrf = subparsers.add_parser("rrf-search",help="ranked search")
    rrf.add_argument("query",type=str,help="Query to search for")
    rrf.add_argument("-k",type=int,default=60,help="Ranking parameter")
    rrf.add_argument("--limit",type=int,default=5,help="Limit search")
    rrf.add_argument("--enhance",type=str,choices=["spell","rewrite","expand"],help="Enhance your search with an LLM")
    rrf.add_argument("--rerank-method",type=str,choices=["individual"],help="Rerank the enhanced search.")
    args = parser.parse_args()

    match args.command:
        case "normalize":
            if args.score:
                with open("data/movies.json", "r") as f:
                    documents = json.load(f)
                search = HybridSearch(documents["movies"])
                search.normalize(args.score)
    
        case "weighted-search":
            with open("data/movies.json", "r") as f:
                documents = json.load(f)
            search = HybridSearch(documents["movies"])

            results = search.weighted_search(args.query,args.alpha,args.limit)

            for i, result in enumerate(results,1):
                doc = result["document"]

                print(f"\n{i}. {doc['title']}")
                print(f"   Hybrid Score: {result['hybrid']:.3f}")
                print(
                    f"   BM25: {result['bm25']:.3f}, "
                    f"Semantic: {result['semantic']:.3f}"
                )
                print(f"   {doc['description'][:100]}...")

        case "rrf-search":
            match args.enhance:
                case "spell":
                    prompt = spell_check(args.query) 
                    query = enhance_query(prompt)
                    print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{query}'\n")
                
                case "rewrite":
                    prompt = rewrite_query(args.query)
                    query = enhance_query(prompt)
                    print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{query}'\n")

                case "expand":
                    prompt = expand_search(args.query)
                    query = enhance_query(prompt)
                    print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{query}'\n") 

                case _:
                    query = args.query
            
            with open("data/movies.json", "r") as f:
                documents = json.load(f)
            search = HybridSearch(documents["movies"])
            results = search.rrf_search(query,args.k,args.limit)
            
            match args.rerank_method:
                case "individual":
                    print(f"\nReranking top {args.limit} results using individual method...")
                    print(f"Reciprocal Rank Fusion Results for '{query}' (k={args.k}):")

                    import time

                    reranked = []

                    for result in results:
                        doc = result["document"]

                        prompt = individual_reranking(query,doc) 
                        
                        score_text = enhance_query(prompt)

                        try:
                            score = float(score_text)
                        except:
                            score = 0.0

                        result["rerank_score"] = score
                        reranked.append(result)

                        time.sleep(3)

                    reranked = sorted(reranked,key=lambda x: x["rerank_score"],reverse=True)

                    final = reranked[:args.limit]

                    for i, result in enumerate(final,1):
                        doc = result["document"]

                        print(f"\n{i}. {doc['title']}")
                        print(f"   Rerank Score: {result['rerank_score']:.3f}/10")
                        print(f"   RRF Score: {result['rrf']:.3f}")
                        print(
                            f"   BM25 Rank: {result['bm25_rank']}, "
                            f"Semantic Rank: {result['semantic_rank']}"
                        )
                        print(f"   {doc['description'][:100]}...")

                case _:
                    print("no rerank method") 
                    for i, result in enumerate(results, 1):
                        doc = result["document"]

                        print(f"\n{i}. {doc['title']}")
                        print(f"   RRF Score: {result['rrf']:.3f}")
                        print(
                            f"   BM25 Rank: {result['bm25_rank']}, "
                            f"Semantic Rank: {result['semantic_rank']}"
                        )
                        print(f"   {doc['description'][:100]}...")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
