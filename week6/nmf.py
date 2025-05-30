#! /usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "joblib",
#     "numpy",
#     "rootpath",
# ]
# ///

import os
import numpy as np
import joblib
import rootpath


def nmf(V, rank, max_iter=500, tol=1e-6, epsilon=1e-10):
    np.random.seed(42)  # do NOT change this.
    # Step 1: Generate initial guesses for W,H
    # Step 2: Initialize the error term. (Hint: ||V-WH||_2)
    # Step 3: For up to max_iter iterations apply:
    ## Step 3.1: The multiplicative update for H
    ## Step 3.2: The multiplicative update for W
    ## Step 3.3: Calculate the current error.
    ## Step 3.4: Check convergence, break if neccessary.
    ## Step 3.5: update the error term
    # Step 4: Return W,H
    pass


def print_top_words(H, feature_names, n_top_words=10):
    for topic_idx, topic in enumerate(H):
        top_indices = topic.argsort()[::-1][:n_top_words]
        top_words = [feature_names[i] for i in top_indices]
        top_scores = topic[top_indices]
        print(f"Topic {topic_idx + 1}:")
        for word, score in zip(top_words, top_scores):
            print(f"  {word} ({score:.3f})")
        print()


def main():
    data_dir = os.path.join(os.path.dirname(__file__), "nmf_data", "train")

    print("Loading data...")
    V = joblib.load(os.path.join(data_dir, "tfidf_matrix.pkl"))
    feature_names = joblib.load(os.path.join(data_dir, "feature_names.pkl"))

    print(f"Data shape: {V.shape}")

    n_topics = 4
    print(f"Running NMF with {n_topics} topics...")
    W, H = nmf(V, rank=n_topics, max_iter=500, tol=1e-6)

    print("\nTop words per topic:")
    print_top_words(H, feature_names, n_top_words=10)


if __name__ == "__main__":
    main()
