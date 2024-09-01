import streamlit as st
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
import uuid

# Initialize the Qdrant client
client = QdrantClient(host="localhost", port=6333)

# Initialize the embedding model
model = SentenceTransformer('all-mpnet-base-v2')

# Create or connect to a Qdrant collection
collection_name = "embeddings_collection"
client.recreate_collection(
    collection_name=collection_name,
    vector_size=768,  # Vector size of the all-mpnet-base-v2 model
    distance="Cosine"
)

# Streamlit UI
st.title("Embedding Storage with Qdrant")
user_input = st.text_input("Enter some text to embed:")

if st.button("Submit"):
    if user_input:
        # Generate embeddings for the input text
        embedding = model.encode(user_input).tolist()

        # Generate a unique ID for this point
        point_id = str(uuid.uuid4())

        # Create a Qdrant PointStruct
        point = PointStruct(
            id=point_id,
            vector=embedding,
            payload={"text": user_input}
        )

        # Store the embedding in the Qdrant database
        client.upsert(
            collection_name=collection_name,
            points=[point]
        )

        st.success(f"Embedding stored in Qdrant with ID: {point_id}")
    else:
        st.warning("Please enter some text.")
