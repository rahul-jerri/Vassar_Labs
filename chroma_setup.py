#METHOD 1
from chromadb import Client
import openai
import os

# Initialize ChromaDB client with persistence
from chromadb import Client

print("Initializing ChromaDB Client...")
client = Client()
print("Client initialized successfully!")

# Additional operations
print("Creating collection...")
collection = client.create_collection("documents")
print("Collection created:", collection.name)


collection.add(
    documents=["Sample document 1", "Sample document 2"],
    metadatas=[{"id": 1}, {"id": 2}],
    ids=["doc1", "doc2"]
)
print("Documents added!")


results = collection.query(
    query_texts=["Sample"],
    n_results=2
)
print("Query Results:", results)

# Set OpenAI API key securely
openai.api_key = os.getenv("api_key")

# Function to generate embeddings
def get_embeddings(text):
    try:
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response['data'][0]['embedding']
    except openai.error.OpenAIError as e:
        print(f"Error during embedding generation: {e}")
        return None


'''

#METHOD 2  WITHOUT APIS TO AVOID QUOTA RESTRICTIONS
from chromadb import Client
from sentence_transformers import SentenceTransformer

# Initialize ChromaDB client with persistence
print("Initializing ChromaDB Client...")
client = Client()
print("Client initialized successfully!")

# Create a collection
print("Creating collection...")
collection = client.create_collection("documents")
print("Collection created:", collection.name)

# Add documents to the collection
collection.add(
    documents=["Sample document 1", "Sample document 2"],
    metadatas=[{"id": 1}, {"id": 2}],
    ids=["doc1", "doc2"]
)
print("Documents added!")

# Initialize sentence-transformers model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to generate embeddings
def get_embeddings(text):
    try:
        embedding = model.encode(text)
        return embedding
    except Exception as e:
        print(f"Error during embedding generation: {e}")
        return None

# Test multiple queries
queries = ["Sample", "document", "sample document"]

# Loop through each query and perform a search in the collection
for query in queries:
    print(f"\nRunning query: {query}")
    
    # Get the embedding for the query
    query_embedding = get_embeddings(query)
    
    if query_embedding is not None:
        # Query the collection with the embedding
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=2
        )
        print(f"Query Results for '{query}':", results)
    else:
        print(f"Skipping query '{query}' due to embedding error.")
'''