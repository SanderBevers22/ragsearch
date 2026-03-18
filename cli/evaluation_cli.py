import argparse
import json

from lib.hybrid_search import HybridSearch

def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    # run evaluation logic here
    with open("data/movies.json", "r") as f:
        documents = json.load(f)
    with open("data/golden_dataset.json", "r") as f:
        golden = json.load(f)
    
    search = HybridSearch(documents["movies"])

    print(f"\nk={limit}\n")

    for case in golden["test_cases"]:
        query = case["query"]
        relevant = case["relevant_docs"]
        relevant_set = set(relevant)
        results = search.rrf_search(query,k=60,limit=limit)
        retrieved_titles = [r["document"]["title"] for r in results]

        top_k = retrieved_titles[:limit]

        relevant_retrieved = sum(1 for title in top_k if title in relevant_set)
        precision = relevant_retrieved/limit
        recall = relevant_retrieved/len(relevant_set)

        if precision == 0 and recall == 0:
            f1 = 0.0
        else:
            f1 = 2*(precision*recall)/(precision+recall)

        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Recall@{limit}: {recall:.4f}")
        print(f"  - F1 Score: {f1}")
        print(f"  - Retrieved: {', '.join(retrieved_titles)}")
        print(f"  - Relevant: {', '.join(relevant)}\n")

if __name__ == "__main__":
    main()
