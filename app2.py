import os
import streamlit as st
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, PayloadSelector, ScoredPoint
import numpy as np

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set up Streamlit page configuration
st.set_page_config(page_title="Qdrant Vector Search", layout="centered", initial_sidebar_state="collapsed")

# Initialize the Qdrant client
client = QdrantClient(host="localhost", port=6333)

# Initialize the embedding model
model = SentenceTransformer('all-mpnet-base-v2')

# Define collection name
collection_name = "embeddings_collection"

# Streamlit UI
st.title("Vector Search with Qdrant")
user_query = st.text_input("Enter your search query:")

if st.button("Search"):
    if user_query:
        # Generate embeddings for the query
        query_embedding = model.encode(user_query).tolist()

        # Perform vector search in Qdrant
        search_results = client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=5  # Return top 5 results
        )

        # Display the search results
        if search_results:
            st.subheader("Top 5 Results:")
            for i, result in enumerate(search_results):
                st.write(f"**Result {i + 1}:**")
                st.write(f"Text: {result.payload['text']}")
                st.write(f"Score: {result.score:.4f}")
                st.write("---")
        else:
            st.warning("No results found.")
    else:
        st.warning("Please enter a search query.")
