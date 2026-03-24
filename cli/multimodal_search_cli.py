import argparse
import json

from lib.multimodal_search import *


def main():
    parser = argparse.ArgumentParser(description="Multimodal search CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify_parser = subparsers.add_parser("verify_image_embedding")
    verify_parser.add_argument("image_path", type=str, help="Path to image file")

    image_parser = subparsers.add_parser("image_search")
    image_parser.add_argument("image_path")

    args = parser.parse_args()

    if args.command == "verify_image_embedding":
        verify_image_embedding(args.image_path)

    elif args.command == "image_search":
        with open("data/movies.json","r") as f:
            documents = json.load(f)
        results = image_search_command(args.image_path, documents["movies"])
        for i, r in enumerate(results, start=1):
            print(f"{i}. {r['title']} (similarity: {r['score']-0.001:.3f})")
            print(f"  {r['description'][:100]}...\n")

if __name__ == "__main__":
    main()
