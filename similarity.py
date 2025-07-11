import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def Alignment(list_to_align, standard_list, adj_matrix):
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    standard_embeddings = model.encode(standard_list)
    to_align_embeddings = model.encode(list_to_align)
    similarity_matrix = cosine_similarity(to_align_embeddings, standard_embeddings)

    best_match_indices = np.argmax(similarity_matrix, axis=1)
    print("best_match_indices:", best_match_indices)

    indexed_pairs = list(enumerate(best_match_indices))
    sorted_pairs = sorted(indexed_pairs, key=lambda x: x[1])
    sorted_indices = [pair[0] for pair in sorted_pairs]

    adj_matrix = np.array(adj_matrix, dtype=int)
    if adj_matrix.ndim != 2:
        raise ValueError("The adjacency matrix must be a two-dimensional array.")

    adjusted_adj = adj_matrix[sorted_indices][:, sorted_indices]

    aligned_activities = [standard_list[pair[1]] for pair in sorted_pairs]
    print("aligned_activities:", aligned_activities)

    return aligned_activities, adjusted_adj