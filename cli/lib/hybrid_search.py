import os

from inverted_index import InvertedIndex
from lib.chunked_semantic_search import ChunkedSemanticSearch

def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score

def rrf_score(rank, k=60):
    return 1 / (k + rank)
    
class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        bm25_results = self._bm25_search(query,limit*500)
        semantic_results = self.semantic_search.search_chunks(query,limit*500)

        bm25_scores = [score for (_,score) in bm25_results]
        semantic_scores = [r["score"] for r in semantic_results]
        
        normalized_bm25 = self.normalize(bm25_scores)
        normalized_semantic = self.normalize(semantic_scores)

        combined = {}

        for (doc_id,raw),norm_score in zip(bm25_results,normalized_bm25):
            docu = next(doc for doc in self.documents if doc["id"] == doc_id)

            combined[doc_id] = {"document": docu, "bm25": norm_score, "semantic": 0.0}

        for result,norm_score in zip(semantic_results,normalized_semantic):
            doc_id = result["id"]

            if doc_id not in combined:
                combined[doc_id] = {"document":result, "bm25":0.0,"semantic":norm_score}
            else:
                combined[doc_id]["semantic"] = norm_score

        for doc_id in combined:
            bm25_score = combined[doc_id]["bm25"]
            semantic_score = combined[doc_id]["semantic"]

            combined[doc_id]["hybrid"] = hybrid_score(bm25_score,semantic_score,alpha)

        sorted_vals = sorted(combined.values(),key=lambda x: x["hybrid"], reverse=True)

        return sorted_vals[:limit]

    def rrf_search(self, query, k, limit=10):
        bm25_results = self._bm25_search(query,limit*500)
        semantic_results = self.semantic_search.search_chunks(query,limit*500)

        combined = {}
        
        for rank,(doc_id,_) in enumerate(bm25_results,start=1):
            document = next(doc for doc in self.documents if doc["id"] == doc_id)
            combined[doc_id] = {"document": document, "bm25_rank": rank, "semantic_rank": None, "rrf":rrf_score(rank,k)}

        for rank,result in enumerate(semantic_results,start=1):
            doc_id = result["id"]
            if doc_id not in combined:
                combined[doc_id] = {"document": result, "bm25_rank": None, "semantic_rank": rank, "rrf":rrf_score(rank,k)}
            else:
                combined[doc_id]["semantic_rank"] = rank
                combined[doc_id]["rrf"]+= rrf_score(rank,k)

        return sorted(combined.values(),key=lambda x: x["rrf"],reverse=True)[:limit]

    def normalize(self,scores):
        max_num = max(scores)
        min_num = min(scores)
        if max_num == min_num:
            for i in range(len(scores)):
                scores[i] = 1.0
        else:
            for i,score in enumerate(scores):
                scores[i] = (score - min_num) / (max_num-min_num)
        return scores
