"""
matching.py

Given one query embedding and a set of candidate embeddings,
finds the best-matching candidate. The one whose embedding is most similar to the query.

Similarity is measured by cosine similarity (how closely two embedding vectors point in the same direction).
This is the step that turns "a set of candidate points" into "one predicted point".
"""

import numpy as np # for handling the candidate point array
import torch # embeddings are torch tensors
import torch.nn.functional as F # provides cosines similarity

# Compare query embedding against a set of candidate embeddings and return the coordinate of the best-matching candidate.
def find_best_match(query_embedding, target_embeddings, candidate_points):
    if len(candidate_points) == 0:
        raise ValueError("find_best_match receives zero candidate points.")
    
    # Add a batch dimension so cosine_similarity can compare the one query against all candidates at once: shape [128] -> [1, 128]
    query_embedding_batch = query_embedding.unsqueeze(0)

    # Cosine similarity between the query and every candidate, one similarity score per candidate
    similarities = F.cosine_similarity(
        query_embedding_batch,
        target_embeddings,
        dim=1
    )

    # The index of the highest similarity score = the best match
    best_index = int(torch.argmax(similarities).item())

    # Return that candidate's coordinate
    return np.asarray(candidate_points[best_index])