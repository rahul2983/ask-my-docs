import streamlit as st
import os
import tempfile
import sqlite3
import uuid
import time
from datetime import datetime
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import torch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get configuration from environment variables
device = os.getenv("DEVICE", "cpu")  # Default to CPU to avoid CUDA errors
db_path = os.getenv("DB_PATH", "pdf_qa.db")
vector_store_dir = os.getenv("VECTOR_STORE_DIR", "vector_stores")
chunk_size = int(os.getenv("CHUNK_SIZE", 1000))
chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 200))
embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
use_openai = os.getenv("USE_OPENAI", "true").lower() == "true"
openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

# Configure the Streamlit page with hamburger menu for better mobile experience
st.set_page_config(
    page_title="Ask My Docs - PDF Q&A", 
    layout="wide",
    menu_items={
        'Get Help': 'https://github.com/yourusername/ask-my-docs',
        'Report a bug': 'https://github.com/yourusername/ask-my-docs/issues',
        'About': 'Ask My Docs - A PDF Q&A application using RAG technology'
    },
    initial_sidebar_state="auto"  # This will adapt to screen size
)

# Initialize database
def init_db():
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS documents (
        id TEXT PRIMARY KEY,
        filename TEXT,
        upload_date TEXT,
        vector_store_path TEXT,
        embedding_type TEXT
    )
    ''')
    conn.commit()
    return conn

# Database operations
def save_document_record(doc_id, filename, vector_store_path, embedding_type="huggingface"):
    conn = init_db()
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO documents VALUES (?, ?, ?, ?, ?)", 
            (doc_id, filename, datetime.now().isoformat(), vector_store_path, embedding_type)
        )
        conn.commit()
    except sqlite3.OperationalError:
        # If the table doesn't have the embedding_type column yet, add it
        c.execute("ALTER TABLE documents ADD COLUMN embedding_type TEXT DEFAULT 'huggingface'")
        conn.commit()
        c.execute(
            "INSERT INTO documents VALUES (?, ?, ?, ?, ?)", 
            (doc_id, filename, datetime.now().isoformat(), vector_store_path, embedding_type)
        )
        conn.commit()
    conn.close()

def get_all_documents():
    conn = init_db()
    c = conn.cursor()
    c.execute("SELECT id, filename, upload_date FROM documents ORDER BY upload_date DESC")
    documents = c.fetchall()
    conn.close()
    return documents

def get_document_by_id(doc_id):
    conn = init_db()
    c = conn.cursor()
    c.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
    document = c.fetchone()
    conn.close()
    return document

def delete_document(doc_id):
    doc = get_document_by_id(doc_id)
    if doc and os.path.exists(doc[3]):
        # Remove the vector store directory
        import shutil
        shutil.rmtree(doc[3])
    
    conn = init_db()
    c = conn.cursor()
    c.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    conn.commit()
    conn.close()

# Add a function to update an existing document or delete and reprocess it
def reprocess_document(doc_id):
    # Get the document record
    doc = get_document_by_id(doc_id)
    if not doc:
        return False, "Document not found."
    
    filename = doc[1]
    old_vector_store_path = doc[3]
    
    # Remove the old vector store if it exists
    if os.path.exists(old_vector_store_path):
        import shutil
        shutil.rmtree(old_vector_store_path)
    
    # Delete the document record from the database
    conn = init_db()
    c = conn.cursor()
    c.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    conn.commit()
    conn.close()
    
    return True, f"Document '{filename}' was removed. Please re-upload it to reprocess with current settings."

# PDF Processing
def process_pdf(pdf_file, filename):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(pdf_file.getvalue())
        temp_path = temp_file.name
    
    # Use PyPDFLoader to extract text from PDF
    loader = PyPDFLoader(temp_path)
    documents = loader.load()
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    
    # Generate a unique ID for this document
    doc_id = str(uuid.uuid4())
    doc_vector_store_path = f"{vector_store_dir}/{doc_id}"
    os.makedirs(vector_store_dir, exist_ok=True)
    
    # Initialize embeddings model (OpenAI or HuggingFace based on settings)
    embedding_type = "openai" if use_openai else "huggingface"
    if use_openai:
        embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    else:
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': device}
        )
    
    # Create vector store
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Save the vector store
    try:
        # Try with newer parameter first
        vector_store.save_local(doc_vector_store_path, allow_dangerous_deserialization=True)
    except TypeError:
        # Fall back to older version without the parameter
        vector_store.save_local(doc_vector_store_path)
    
    # Save document record in database with embedding type
    save_document_record(doc_id, filename, doc_vector_store_path, embedding_type)
    
    # Clean up
    os.unlink(temp_path)
    
    return doc_id

# Query Processing
def query_document(doc_id, query):
    # Get document record
    doc = get_document_by_id(doc_id)
    if not doc:
        return "Document not found."
    
    vector_store_path = doc[3]
    filename = doc[1]
    
    # Get the embedding type from the document record
    embedding_type = "huggingface"  # Default for compatibility with older records
    if len(doc) > 4:  # If the embedding_type column exists
        embedding_type = doc[4]
    
    # Initialize embeddings model based on the document's embedding type
    if embedding_type == "openai":
        embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",  # Consistent model for OpenAI
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    else:
        # Use Hugging Face embeddings for older documents or if that's what was used
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",  # Consistent model for HF
            model_kwargs={'device': device}
        )
    
    try:
        # Load the vector store with the appropriate embeddings - handle both versions
        try:
            # Try with newer parameter first
            vector_store = FAISS.load_local(
                vector_store_path, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
        except TypeError:
            # Fall back to older version without the parameter
            vector_store = FAISS.load_local(
                vector_store_path, 
                embeddings
            )
        
        # Set up the language model (OpenAI's ChatGPT)
        from langchain_openai import ChatOpenAI
        from langchain.schema import SystemMessage, HumanMessage
        
        llm = ChatOpenAI(
            model_name=openai_model,
            temperature=0.7,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Create a basic retrieval chain without custom prompts
        retriever = vector_store.as_retriever(
            search_kwargs={"k": 5}  # Retrieve 5 chunks for good coverage
        )
        
        # Get documents from the retriever
        retrieved_docs = retriever.get_relevant_documents(query)
        
        # Extract content from documents to form context
        doc_content = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # Create a system prompt for ChatGPT
        system_prompt = f"""You are an AI assistant answering questions about a document titled '{filename}'.
        
Here are excerpts from the document:

{doc_content}

Based on these excerpts, please answer the following question. DO NOT mention that you're only seeing excerpts.
Answer as if you've read the entire document. DO NOT say things like "Based on the provided excerpts" or 
"I don't have access to the full document." Just answer directly based on the information you have.
"""
        # Use the chat model directly
        user_message = query
        
        # Get response from ChatGPT
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]
        
        response = llm.invoke(messages)
        
        # Return the response and retrieved documents
        return {
            "answer": response.content,
            "source_documents": retrieved_docs
        }
    except AssertionError as e:
        if "assert d == self.d" in str(e):
            # This is a dimension mismatch error
            return {
                "answer": "There was a dimension mismatch error when querying this document. This usually happens when the document was embedded with a different model than what is currently being used. Please re-upload the document to fix this issue.",
                "source_documents": []
            }
        else:
            raise e

# Streamlit UI
def main():
    st.title("Ask My Docs - PDF Q&A")
    st.markdown(
        """
        Upload your PDFs and ask questions about their content. This app uses 
        retrieval-augmented generation (RAG) to provide accurate answers based on your documents.
        """
    )
    
    # Initialize the database
    init_db()
    
    # Add a Home button to the top right
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("🏠 Home", help="Return to home page", use_container_width=True):
            # Clear session state to reset the app
            if 'selected_doc' in st.session_state:
                del st.session_state['selected_doc']
            if 'selected_doc_name' in st.session_state:
                del st.session_state['selected_doc_name']
            st.rerun()
    
    # Check if we need to close the sidebar after selecting a document
    if 'close_sidebar' in st.session_state and st.session_state.close_sidebar:
        # Reset the flag
        st.session_state.close_sidebar = False
        # Use query_params to collapse sidebar
        st.query_params["sidebar"] = "collapsed"
    
    # Sidebar for document management
    with st.sidebar:
        st.header("Document Management")
        
        # Upload new document
        uploaded_file = st.file_uploader("Upload a PDF document", type="pdf", key="pdf_uploader")
        if uploaded_file is not None:
            try:
                # Function to process a new document
                def process_new_document():
                    with st.spinner("Processing PDF... This may take a moment."):
                        doc_id = process_pdf(uploaded_file, uploaded_file.name)
                        st.success(f"Document processed successfully!", icon="✅")
                        # Set this as the selected document
                        st.session_state['selected_doc'] = doc_id
                        st.session_state['selected_doc_name'] = uploaded_file.name
                        # Set flag to close sidebar after rerun
                        st.session_state.close_sidebar = True
                        # Force a rerun to refresh the document list
                        st.rerun()
                
                # Check if a document with this name already exists to avoid duplicates
                documents = get_all_documents()
                existing_filenames = [doc[1] for doc in documents]
                
                # Find exact filename match (case-sensitive)
                exact_match = any(filename == uploaded_file.name for filename in existing_filenames)
                
                if not exact_match:
                    # New document, process it
                    process_new_document()
                else:
                    # Document with same name exists, ask what to do
                    st.info(f"A document with name '{uploaded_file.name}' already exists.")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Use existing document", key="use_existing"):
                            # If document already exists, just select it
                            existing_doc = next((doc for doc in documents if doc[1] == uploaded_file.name), None)
                            if existing_doc:
                                st.session_state['selected_doc'] = existing_doc[0]
                                st.session_state['selected_doc_name'] = existing_doc[1]
                                # Set flag to close sidebar after rerun
                                st.session_state.close_sidebar = True
                                st.success(f"Using existing document '{uploaded_file.name}'", icon="✅")
                                st.rerun()
                    with col2:
                        if st.button("Process as new document", key="process_new"):
                            process_new_document()
            except Exception as e:
                st.error(f"Error processing document: {str(e)}", icon="🚨")
                if "CUDA" in str(e) or "GPU" in str(e):
                    st.warning("""
                    CUDA/GPU error detected. Please make sure:
                    1. Your DEVICE is set to 'cpu' in the .env file
                    2. You're using a compatible version of PyTorch
                    
                    Try restarting the application after making these changes.
                    """, icon="⚠️")
                    # Log the full error for debugging
                    st.exception(e)
        
        st.divider()
        
        # List existing documents
        st.subheader("Your Documents")
        documents = get_all_documents()
        
        if not documents:
            st.info("No documents uploaded yet.", icon="ℹ️")
        
        # Keep track of which documents we've already displayed to avoid duplicates
        displayed_docs = set()
        
        # Make button layout more mobile-friendly
        for doc in documents:
            # Skip if we've already displayed this document (avoid duplicates)
            if doc[0] in displayed_docs:
                continue
                
            displayed_docs.add(doc[0])
            
            # More mobile-friendly layout - stack buttons vertically on small screens
            doc_col = st.container()
            with doc_col:
                if st.button(f"📄 {doc[1]}", key=f"select_{doc[0]}", use_container_width=True):
                    st.session_state['selected_doc'] = doc[0]
                    st.session_state['selected_doc_name'] = doc[1]
                    # Set flag to close sidebar after rerun
                    st.session_state.close_sidebar = True
                    st.rerun()
                
                action_cols = st.columns(2)
                with action_cols[0]:
                    # Add reprocess button
                    if st.button("🔄 Reprocess", key=f"reprocess_{doc[0]}", 
                                help="Reprocess this document with current settings",
                                use_container_width=True):
                        success, message = reprocess_document(doc[0])
                        if success:
                            st.success(message, icon="✅")
                            if 'selected_doc' in st.session_state and st.session_state['selected_doc'] == doc[0]:
                                del st.session_state['selected_doc']
                            st.rerun()
                        else:
                            st.error(message, icon="🚨")
                with action_cols[1]:
                    if st.button("🗑️ Delete", key=f"delete_{doc[0]}", 
                                help="Delete this document",
                                use_container_width=True):
                        delete_document(doc[0])
                        if 'selected_doc' in st.session_state and st.session_state['selected_doc'] == doc[0]:
                            del st.session_state['selected_doc']
                        st.rerun()
                
                # Add a small divider between documents
                st.markdown("---")
    
    # Main content area
    if 'selected_doc' in st.session_state and st.session_state['selected_doc']:
        st.header(f"Querying: {st.session_state['selected_doc_name']}")
        
        # Query input - larger and more visible
        query = st.text_input("Ask a question about your document:", 
                             key="query_input", 
                             help="Type your question here and press Enter")
        
        # Add some quick query suggestions for mobile users
        st.markdown("**Quick queries:**")
        suggestion_cols = st.columns(2)
        with suggestion_cols[0]:
            if st.button("📋 Summarize document", key="summarize_btn", use_container_width=True):
                # Instead of directly modifying session_state, we'll process this query immediately
                with st.spinner("Processing document summary..."):
                    # Create a summarization query
                    summary_query = "Summarize this document"
                    result = query_document(st.session_state['selected_doc'], summary_query)
                    
                    if isinstance(result, dict) and "answer" in result:
                        st.subheader("Summary")
                        st.write(result["answer"])
                        
                        if result["source_documents"]:
                            with st.expander("View Source Context"):
                                for i, doc in enumerate(result["source_documents"]):
                                    st.markdown(f"**Source {i+1}**")
                                    st.markdown(f"Page: {doc.metadata.get('page', 'N/A')}")
                                    st.markdown(f"```\n{doc.page_content}\n```")
        with suggestion_cols[1]:
            if st.button("❓ Key points", key="keypoints_btn", use_container_width=True):
                # Process key points query immediately
                with st.spinner("Finding key points..."):
                    # Create a key points query
                    key_points_query = "What are the key points?"
                    result = query_document(st.session_state['selected_doc'], key_points_query)
                    
                    if isinstance(result, dict) and "answer" in result:
                        st.subheader("Key Points")
                        st.write(result["answer"])
                        
                        if result["source_documents"]:
                            with st.expander("View Source Context"):
                                for i, doc in enumerate(result["source_documents"]):
                                    st.markdown(f"**Source {i+1}**")
                                    st.markdown(f"Page: {doc.metadata.get('page', 'N/A')}")
                                    st.markdown(f"```\n{doc.page_content}\n```")
        
        if query:
            try:
                with st.spinner("Processing your question about the document..."):
                    result = query_document(st.session_state['selected_doc'], query)
                    
                    if isinstance(result, dict) and "answer" in result:
                        st.subheader("Answer")
                        
                        # Check if this is an error message about dimension mismatch
                        if "dimension mismatch error" in result["answer"]:
                            st.error(result["answer"], icon="🚨")
                            st.warning("""
                            This error occurs when you try to query a document that was embedded with a different model.
                            This typically happens when switching between Hugging Face and OpenAI embeddings.
                            
                            To fix this issue:
                            1. Click the 🔄 button next to the document to remove it
                            2. Re-upload the document to process it with the current settings
                            """, icon="⚠️")
                        else:
                            st.write(result["answer"])
                            
                            if result["source_documents"]:
                                with st.expander("View Source Context"):
                                    for i, doc in enumerate(result["source_documents"]):
                                        st.markdown(f"**Source {i+1}**")
                                        st.markdown(f"Page: {doc.metadata.get('page', 'N/A')}")
                                        st.markdown(f"```\n{doc.page_content}\n```")
                    else:
                        st.error("An unexpected error occurred. Please try again.", icon="🚨")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}", icon="🚨")
                if "assert d == self.d" in str(e):
                    st.warning("""
                    Dimension mismatch error detected. This happens when a document was embedded with a different model.
                    
                    To fix this issue:
                    1. Click the 🔄 button next to the document in the sidebar
                    2. Re-upload the document to process it with current settings
                    """, icon="⚠️")
                elif "CUDA" in str(e) or "GPU" in str(e):
                    st.warning("""
                    CUDA/GPU error detected. Please make sure:
                    1. Your DEVICE is set to 'cpu' in the .env file
                    2. You're using a compatible version of PyTorch
                    
                    The application will continue to work on CPU, which may be slower but will be compatible with all systems.
                    """, icon="⚠️")
                # Log the full error for debugging
                st.exception(e)
        
        # Add a "Choose another document" button at the bottom when viewing results
        st.divider()
        if st.button("📚 Choose another document", key="choose_another_doc", use_container_width=True):
            if 'selected_doc' in st.session_state:
                del st.session_state['selected_doc']
            if 'selected_doc_name' in st.session_state:
                del st.session_state['selected_doc_name']
            st.rerun()
    else:
        st.info("👈 Please upload or select a document from the sidebar to get started.", icon="ℹ️")
        # Add more helpful mobile instructions
        st.markdown("""
        ### Getting Started:
        1. Tap the **≡** menu icon in the top-left corner
        2. Upload your PDF in the sidebar that appears
        3. Your document will be processed and ready for questions
        """)

if __name__ == "__main__":
    main()