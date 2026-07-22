"""
find_best_match.py

You give it one query (the click) and a set of candidate points, and it finds the single candidate that looks like the query.
The best match is measured by cosine similarity, a number saying how closely two embeddings point in the same direction.
"""

import numpy as np # for handling the candidate positions
import torch # embeddings are torch sensors
import torch.nn.functional as F # this gives us cosine similarity

# find the candidate that best matches the query.
def find_best_match(query_embedding, target_embeddings, candidate_points):
    if len(candidate_points) == 0:
        raise ValueError("find_best_match received zero candidate points.")
    
    # wrap query_embedding so it can be compared against all candidates in one go
    query_embedding_batch = query_embedding.unsqueeze(0)

    # gives one similarity score per candidate
    similarities = F.cosine_similarity(
        query_embedding_batch,
        target_embeddings,
        dim=1,
    )

    # find the position of the highest score, that's the best match
    best_index = int(torch.argmax(similarities).item())

    return np.asarray(candidate_points[best_index])