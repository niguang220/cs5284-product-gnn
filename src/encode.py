"""Encode node text into sentence embeddings with a frozen encoder."""
import numpy as np
from sentence_transformers import SentenceTransformer

# Pinned to a commit of the Hugging Face model repo so everyone gets the same weights.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def load_encoder(device="cpu"):
    return SentenceTransformer(MODEL_NAME, revision=MODEL_REVISION, device=device)


def encode(texts, encoder=None, batch_size=64, show_progress_bar=True):
    """Return a float32 array of shape [len(texts), 384]; row i embeds texts[i].

    Texts longer than the encoder's max_seq_length (256 tokens for this model) are truncated.
    """
    encoder = encoder or load_encoder()
    emb = encoder.encode(
        list(texts), batch_size=batch_size, show_progress_bar=show_progress_bar, convert_to_numpy=True
    )
    return emb.astype(np.float32)
