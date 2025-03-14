import streamlit as st
import pinecone
import openai
from pypdf import PdfReader
import tempfile

# Set API Keys (Replace with your actual keys)
OPENAI_API_KEY = "your_openai_api_key"
PINECONE_API_KEY = "your_pinecone_api_key"
PINECONE_ENV = "your_pinecone_environment"

# Initialize OpenAI & Pinecone
openai.api_key = OPENAI_API_KEY
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# Create Pinecone Index if not exists
index_name = "finance-insights"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(name=index_name, dimension=1536, metric="cosine")

index = pinecone.Index(index_name)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

# Function to get text embeddings
def get_embedding(text):
    response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
    return response["data"][0]["embedding"]

# Function to store PDF in Pinecone
def store_pdf_in_pinecone(pdf_file, pdf_name):
    text = extract_text_from_pdf(pdf_file)
    embedding = get_embedding(text)
    index.upsert([(pdf_name, embedding, {"filename": pdf_name})])
    return f"Stored {pdf_name} in Pinecone."

# Function to search PDFs in Pinecone
def search_pdfs(query):
    query_vector = get_embedding(query)
    results = index.query(vector=query_vector, top_k=2, include_metadata=True)
    return results["matches"]

# Streamlit Web App UI
st.title("📄 AI-Powered PDF Search (Finance & Insurance)")
st.write("Upload your **insurance policies, mutual fund reports, or tax documents**, and ask AI-powered questions.")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        pdf_path = temp_file.name
        pdf_name = uploaded_file.name

    st.success(f"Uploaded: {pdf_name}")
    if st.button("Process & Store in Pinecone"):
        msg = store_pdf_in_pinecone(pdf_path, pdf_name)
        st.success(msg)

# Search Query Input
query = st.text_input("🔎 Ask a question about your documents:")
if query:
    results = search_pdfs(query)
    if results:
        st.subheader("📌 Relevant Results:")
        for match in results:
            st.write(f"**File:** {match['metadata']['filename']} (Score: {match['score']:.2f})")
    else:
        st.warning("No relevant documents found.")