# --- HELPER: RAG FUSION ALGORITHM ---
def reciprocal_rank_fusion(results: list[list], k=60):
    """ Reciprocal Rank Fusion that fuses multiple lists of ranked documents. """
    fused_scores = {}
    doc_map = {}

    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = doc.page_content
            doc_map[doc_str] = doc
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort by score descending
    reranked_results = [
        (doc_map[doc], score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    # Return top 5 unique documents
    return [doc for doc, _ in reranked_results[:5]]
