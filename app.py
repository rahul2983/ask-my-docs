import streamlit as st
import os
import tempfile
import sqlite3
import uuid
import time
import hashlib
import hmac
import secrets
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

# Secret key for password hashing - would be better to store this in .env file
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-for-password-hashing")

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

# Initialize database with additional tables for users
def init_db():
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Check if documents table exists already
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='documents'")
    documents_table_exists = c.fetchone() is not None
    
    # Original documents table
    c.execute('''
    CREATE TABLE IF NOT EXISTS documents (
        id TEXT PRIMARY KEY,
        filename TEXT,
        upload_date TEXT,
        vector_store_path TEXT,
        embedding_type TEXT
    )
    ''')
    
    # Add user_id column to documents table if it doesn't exist
    if documents_table_exists:
        # Check if user_id column exists
        c.execute("PRAGMA table_info(documents)")
        columns = c.fetchall()
        column_names = [column[1] for column in columns]
        
        if 'user_id' not in column_names:
            # Add user_id column
            c.execute("ALTER TABLE documents ADD COLUMN user_id TEXT")
            # Set existing documents to belong to a default user
            # Change this default_user_id if you want to assign to a specific user
            default_user_id = "default_user"
            c.execute("UPDATE documents SET user_id = ?", (default_user_id,))
            conn.commit()
    
    # New users table
    c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        username TEXT UNIQUE,
        email TEXT UNIQUE,
        password_hash TEXT,
        created_at TEXT
    )
    ''')
    
    # Table for password reset tokens
    c.execute('''
    CREATE TABLE IF NOT EXISTS password_reset_tokens (
        user_id TEXT,
        token TEXT,
        created_at TEXT,
        expires_at TEXT,
        PRIMARY KEY (user_id, token)
    )
    ''')
    
    # Create a default user if needed (for existing documents)
    if documents_table_exists:
        c.execute("SELECT COUNT(*) FROM users WHERE id = ?", ("default_user",))
        default_user_exists = c.fetchone()[0] > 0
        
        if not default_user_exists:
            # Create a default user for existing documents
            c.execute(
                "INSERT INTO users VALUES (?, ?, ?, ?, ?)",
                ("default_user", "Default User", "default@example.com", 
                 hash_password("change_this_password"), datetime.now().isoformat())
            )
            conn.commit()
    
    conn.commit()
    return conn

# User authentication functions
def hash_password(password):
    """Hash a password using HMAC and SHA-256"""
    return hmac.new(
        SECRET_KEY.encode('utf-8'),
        password.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

def create_user(username, email, password):
    """Create a new user in the database"""
    conn = init_db()
    c = conn.cursor()
    user_id = str(uuid.uuid4())
    password_hash = hash_password(password)
    created_at = datetime.now().isoformat()
    
    try:
        c.execute(
            "INSERT INTO users VALUES (?, ?, ?, ?, ?)",
            (user_id, username, email, password_hash, created_at)
        )
        conn.commit()
        conn.close()
        return True, user_id
    except sqlite3.IntegrityError:
        conn.close()
        # Check if username or email already exists
        conn = init_db()
        c = conn.cursor()
        c.execute("SELECT id FROM users WHERE username = ?", (username,))
        username_exists = c.fetchone() is not None
        
        c.execute("SELECT id FROM users WHERE email = ?", (email,))
        email_exists = c.fetchone() is not None
        conn.close()
        
        if username_exists:
            return False, "Username already exists"
        elif email_exists:
            return False, "Email already exists"
        else:
            return False, "An error occurred during registration"

def verify_user(username, password):
    """Verify user credentials"""
    conn = init_db()
    c = conn.cursor()
    c.execute("SELECT id, password_hash FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    conn.close()
    
    if user and user[1] == hash_password(password):
        return True, user[0]
    return False, None

def create_password_reset_token(user_id):
    """Create a password reset token for a user"""
    conn = init_db()
    c = conn.cursor()
    token = secrets.token_urlsafe(32)
    created_at = datetime.now().isoformat()
    # Token expires in 1 hour
    expires_at = (datetime.now() + datetime.timedelta(hours=1)).isoformat()
    
    c.execute(
        "INSERT INTO password_reset_tokens VALUES (?, ?, ?, ?)",
        (user_id, token, created_at, expires_at)
    )
    conn.commit()
    conn.close()
    return token

def verify_reset_token(token):
    """Verify a password reset token"""
    conn = init_db()
    c = conn.cursor()
    c.execute(
        "SELECT user_id, expires_at FROM password_reset_tokens WHERE token = ?", 
        (token,)
    )
    result = c.fetchone()
    conn.close()
    
    if not result:
        return False, None
    
    user_id, expires_at = result
    if datetime.fromisoformat(expires_at) < datetime.now():
        return False, None
    
    return True, user_id

def reset_password(user_id, new_password):
    """Reset a user's password"""
    conn = init_db()
    c = conn.cursor()
    password_hash = hash_password(new_password)
    
    c.execute(
        "UPDATE users SET password_hash = ? WHERE id = ?",
        (password_hash, user_id)
    )
    
    # Delete all password reset tokens for this user
    c.execute(
        "DELETE FROM password_reset_tokens WHERE user_id = ?",
        (user_id,)
    )
    
    conn.commit()
    conn.close()
    return True

def get_user_by_email(email):
    """Get a user by email"""
    conn = init_db()
    c = conn.cursor()
    c.execute("SELECT id, username FROM users WHERE email = ?", (email,))
    user = c.fetchone()
    conn.close()
    return user

def get_user_by_id(user_id):
    """Get a user by ID"""
    conn = init_db()
    c = conn.cursor()
    c.execute("SELECT id, username, email FROM users WHERE id = ?", (user_id,))
    user = c.fetchone()
    conn.close()
    return user

# Modified database operations to include user_id
def save_document_record(doc_id, filename, vector_store_path, user_id=None, embedding_type="huggingface"):
    conn = init_db()
    c = conn.cursor()
    try:
        # Check if user_id column exists
        c.execute("PRAGMA table_info(documents)")
        columns = c.fetchall()
        column_names = [column[1] for column in columns]
        
        if 'user_id' in column_names:
            # If user_id column exists, include it in the query
            c.execute(
                "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?)", 
                (doc_id, filename, datetime.now().isoformat(), vector_store_path, embedding_type, user_id)
            )
        else:
            # If user_id column doesn't exist yet, use the old query format
            c.execute(
                "INSERT INTO documents VALUES (?, ?, ?, ?, ?)", 
                (doc_id, filename, datetime.now().isoformat(), vector_store_path, embedding_type)
            )
        conn.commit()
    except sqlite3.OperationalError as e:
        # Try to handle errors more gracefully
        if "no such column: embedding_type" in str(e):
            # Add embedding_type column if it doesn't exist yet
            c.execute("ALTER TABLE documents ADD COLUMN embedding_type TEXT DEFAULT 'huggingface'")
            conn.commit()
            # Try again with the right columns
            save_document_record(doc_id, filename, vector_store_path, user_id, embedding_type)
        else:
            # For other errors, log and re-raise
            print(f"Database error: {e}")
            raise e
    conn.close()

def get_user_documents(user_id):
    """Get documents for a specific user"""
    conn = init_db()
    c = conn.cursor()
    
    # Check if user_id column exists
    c.execute("PRAGMA table_info(documents)")
    columns = c.fetchall()
    column_names = [column[1] for column in columns]
    
    if 'user_id' in column_names:
        # If user_id column exists, filter by it
        c.execute(
            "SELECT id, filename, upload_date FROM documents WHERE user_id = ? ORDER BY upload_date DESC", 
            (user_id,)
        )
    else:
        # If user_id column doesn't exist, return all documents
        # This is just for backward compatibility
        c.execute("SELECT id, filename, upload_date FROM documents ORDER BY upload_date DESC")
        
    documents = c.fetchall()
    conn.close()
    return documents

def get_document_by_id(doc_id, user_id=None):
    """Get document by ID, optionally checking if it belongs to a specific user"""
    conn = init_db()
    c = conn.cursor()
    
    # Check if user_id column exists
    c.execute("PRAGMA table_info(documents)")
    columns = c.fetchall()
    column_names = [column[1] for column in columns]
    
    if 'user_id' in column_names and user_id:
        # Get document only if it belongs to the user
        c.execute("SELECT * FROM documents WHERE id = ? AND user_id = ?", (doc_id, user_id))
    else:
        # Get document regardless of user
        c.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
        
    document = c.fetchone()
    conn.close()
    return document

def delete_document(doc_id, user_id=None):
    """Delete a document, optionally checking if it belongs to a specific user"""
    # Get the document record
    doc = get_document_by_id(doc_id, user_id)
    if not doc:
        return False, "Document not found or you don't have permission to delete it."
    
    if doc and os.path.exists(doc[3]):
        # Remove the vector store directory
        import shutil
        shutil.rmtree(doc[3])
    
    conn = init_db()
    c = conn.cursor()
    
    # Check if user_id column exists
    c.execute("PRAGMA table_info(documents)")
    columns = c.fetchall()
    column_names = [column[1] for column in columns]
    
    if 'user_id' in column_names and user_id:
        # Delete document only if it belongs to the user
        c.execute("DELETE FROM documents WHERE id = ? AND user_id = ?", (doc_id, user_id))
    else:
        # Delete document regardless of user
        c.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        
    conn.commit()
    conn.close()
    return True, "Document deleted successfully."

def reprocess_document(doc_id, user_id=None):
    """Reprocess a document, optionally checking if it belongs to a specific user"""
    # Get the document record
    doc = get_document_by_id(doc_id, user_id)
    if not doc:
        return False, "Document not found or you don't have permission to reprocess it."
    
    filename = doc[1]
    old_vector_store_path = doc[3]
    
    # Remove the old vector store if it exists
    if os.path.exists(old_vector_store_path):
        import shutil
        shutil.rmtree(old_vector_store_path)
    
    # Delete the document record from the database
    conn = init_db()
    c = conn.cursor()
    
    # Check if user_id column exists
    c.execute("PRAGMA table_info(documents)")
    columns = c.fetchall()
    column_names = [column[1] for column in columns]
    
    if 'user_id' in column_names and user_id:
        # Delete document only if it belongs to the user
        c.execute("DELETE FROM documents WHERE id = ? AND user_id = ?", (doc_id, user_id))
    else:
        # Delete document regardless of user
        c.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        
    conn.commit()
    conn.close()
    
    return True, f"Document '{filename}' was removed. Please re-upload it to reprocess with current settings."

# PDF Processing
def process_pdf(pdf_file, filename, user_id):
    """Process a PDF file and associate it with a user"""
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
    
    # Save document record in database with embedding type and user_id
    save_document_record(doc_id, filename, doc_vector_store_path, user_id, embedding_type)
    
    # Clean up
    os.unlink(temp_path)
    
    return doc_id

# Query Processing (unchanged)
def query_document(doc_id, query, user_id=None):
    """Query a document, optionally checking if it belongs to a specific user"""
    # Get document record, optionally checking user ownership
    doc = get_document_by_id(doc_id, user_id)
    if not doc:
        return "Document not found or you don't have permission to access it."
    
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

# Session state helper functions
def init_session_state():
    """Initialize session state variables for authentication"""
    if 'user_id' not in st.session_state:
        st.session_state['user_id'] = None
    if 'username' not in st.session_state:
        st.session_state['username'] = None
    if 'is_authenticated' not in st.session_state:
        st.session_state['is_authenticated'] = False
    if 'auth_page' not in st.session_state:
        st.session_state['auth_page'] = 'login'  # Options: login, register, forgot_password, reset_password

def login_user(user_id, username):
    """Set session state for a logged-in user"""
    st.session_state['user_id'] = user_id
    st.session_state['username'] = username
    st.session_state['is_authenticated'] = True

def logout_user():
    """Clear session state for a logged-out user"""
    st.session_state['user_id'] = None
    st.session_state['username'] = None
    st.session_state['is_authenticated'] = False
    # Clear selected document if any
    if 'selected_doc' in st.session_state:
        del st.session_state['selected_doc']
    if 'selected_doc_name' in st.session_state:
        del st.session_state['selected_doc_name']

# UI Components for Authentication
def render_login_page():
    """Render the login page"""
    st.header("Login")
    
    with st.form("login_form"):
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        submit_button = st.form_submit_button("Login", use_container_width=True)
        
        if submit_button:
            if not username or not password:
                st.error("Please enter both username and password.")
            else:
                success, user_id = verify_user(username, password)
                if success:
                    login_user(user_id, username)
                    st.success("Login successful!")
                    time.sleep(1)  # Small delay for success message
                    st.rerun()  # Refresh the page
                else:
                    st.error("Invalid username or password.")
    
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Register New Account", key="goto_register", use_container_width=True):
            st.session_state['auth_page'] = 'register'
            st.rerun()
    with col2:
        if st.button("Forgot Password?", key="goto_forgot_password", use_container_width=True):
            st.session_state['auth_page'] = 'forgot_password'
            st.rerun()

def render_register_page():
    """Render the registration page"""
    st.header("Create an Account")
    
    with st.form("register_form"):
        username = st.text_input("Username", key="register_username")
        email = st.text_input("Email", key="register_email")
        password = st.text_input("Password", type="password", key="register_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="register_confirm_password")
        submit_button = st.form_submit_button("Register", use_container_width=True)
        
        if submit_button:
            if not username or not email or not password or not confirm_password:
                st.error("Please fill out all fields.")
            elif password != confirm_password:
                st.error("Passwords do not match.")
            elif len(password) < 8:
                st.error("Password must be at least 8 characters long.")
            elif '@' not in email or '.' not in email:
                st.error("Please enter a valid email address.")
            else:
                success, result = create_user(username, email, password)
                if success:
                    # Auto-login after successful registration
                    login_user(result, username)
                    st.success("Registration successful! Logging you in...")
                    time.sleep(1)  # Small delay for success message
                    st.rerun()
                else:
                    st.error(f"Registration failed: {result}")
    
    st.divider()
    if st.button("Back to Login", key="goto_login_from_register", use_container_width=True):
        st.session_state['auth_page'] = 'login'
        st.rerun()

def render_forgot_password_page():
    """Render the forgot password page"""
    st.header("Reset Your Password")
    st.write("Enter your email address to receive a password reset link.")
    
    with st.form("forgot_password_form"):
        email = st.text_input("Email Address", key="forgot_password_email")
        submit_button = st.form_submit_button("Send Reset Instructions", use_container_width=True)
        
        if submit_button:
            if not email:
                st.error("Please enter your email address.")
            else:
                user = get_user_by_email(email)
                if user:
                    user_id, username = user
                    token = create_password_reset_token(user_id)
                    # In a real application, you would send an email with the reset link
                    # For this demo, we'll show the token directly
                    st.success(f"Password reset instructions sent to {email}!")
                    st.info(f"Your password reset token is: {token}")
                    st.info("In a real application, this would be sent via email.")
                else:
                    # Still show success to prevent email enumeration
                    st.success(f"If an account exists with the email {email}, password reset instructions have been sent.")
    
    st.divider()
    if st.button("Back to Login", key="goto_login_from_forgot", use_container_width=True):
        st.session_state['auth_page'] = 'login'
        st.rerun()

def render_reset_password_page():
    """Render the reset password page"""
    st.header("Set a New Password")
    
    with st.form("reset_password_form"):
        token = st.text_input("Reset Token", key="reset_token")
        new_password = st.text_input("New Password", type="password", key="new_password")
        confirm_password = st.text_input("Confirm New Password", type="password", key="confirm_new_password")
        submit_button = st.form_submit_button("Reset Password", use_container_width=True)
        
        if submit_button:
            if not token or not new_password or not confirm_password:
                st.error("Please fill out all fields.")
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            elif len(new_password) < 8:
                st.error("Password must be at least 8 characters long.")
            else:
                valid, user_id = verify_reset_token(token)
                if valid:
                    success = reset_password(user_id, new_password)
                    if success:
                        st.success("Password reset successful! You can now login with your new password.")
                        time.sleep(2)  # Small delay for success message
                        st.session_state['auth_page'] = 'login'
                        st.rerun()
                else:
                    st.error("Invalid or expired token. Please request a new password reset link.")
    
    st.divider()
    if st.button("Back to Login", key="goto_login_from_reset", use_container_width=True):
        st.session_state['auth_page'] = 'login'
        st.rerun()

def render_user_menu():
    """Render user menu for logged-in users"""
    # Account dropdown in the top corner
    with st.container():
        col1, col2, col3 = st.columns([1, 4, 2])
        with col3:
            menu_option = st.selectbox(
                f"👤 {st.session_state['username']}",
                ["My Account", "Logout"],
                label_visibility="collapsed"
            )
            
            if menu_option == "Logout":
                logout_user()
                st.rerun()
            elif menu_option == "My Account":
                # Could add account settings page here in the future
                pass

# Streamlit UI
def main():
    # Initialize session state for authentication
    init_session_state()
    
    # Initialize the database
    init_db()
    
    # Check if user is authenticated
    if not st.session_state['is_authenticated']:
        st.title("Ask My Docs - PDF Q&A")
        
        # Render the appropriate authentication page
        if st.session_state['auth_page'] == 'login':
            render_login_page()
        elif st.session_state['auth_page'] == 'register':
            render_register_page()
        elif st.session_state['auth_page'] == 'forgot_password':
            render_forgot_password_page()
        elif st.session_state['auth_page'] == 'reset_password':
            render_reset_password_page()
    else:
        # User is authenticated - show the main application
        st.title("Ask My Docs - PDF Q&A")
        
        # Render user menu
        render_user_menu()
        
        st.markdown(
            """
            Upload your PDFs and ask questions about their content. This app uses 
            retrieval-augmented generation (RAG) to provide accurate answers based on your documents.
            """
        )
        
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
                            # Associate the document with the current user
                            doc_id = process_pdf(uploaded_file, uploaded_file.name, st.session_state['user_id'])
                            st.success(f"Document processed successfully!", icon="✅")
                            # Set this as the selected document
                            st.session_state['selected_doc'] = doc_id
                            st.session_state['selected_doc_name'] = uploaded_file.name
                            # Set flag to close sidebar after rerun
                            st.session_state.close_sidebar = True
                            # Force a rerun to refresh the document list
                            st.rerun()
                    
                    # Check if a document with this name already exists for this user
                    documents = get_user_documents(st.session_state['user_id'])
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
            
            # List existing documents for the current user
            st.subheader("Your Documents")
            user_documents = get_user_documents(st.session_state['user_id'])
            
            if not user_documents:
                st.info("No documents uploaded yet.", icon="ℹ️")
            
            # Keep track of which documents we've already displayed to avoid duplicates
            displayed_docs = set()
            
            # Make button layout more mobile-friendly
            for doc in user_documents:
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
                            success, message = reprocess_document(doc[0], st.session_state['user_id'])
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
                            success, message = delete_document(doc[0], st.session_state['user_id'])
                            if success:
                                if 'selected_doc' in st.session_state and st.session_state['selected_doc'] == doc[0]:
                                    del st.session_state['selected_doc']
                                st.rerun()
                            else:
                                st.error(message, icon="🚨")
                    
                    # Add a small divider between documents
                    st.markdown("---")
        
        # Main content area
        if 'selected_doc' in st.session_state and st.session_state['selected_doc']:
            # Verify the document belongs to the current user
            doc = get_document_by_id(st.session_state['selected_doc'], st.session_state['user_id'])
            if not doc:
                st.error("Document not found or you don't have permission to access it.", icon="🚨")
                # Clear selected document
                del st.session_state['selected_doc']
                if 'selected_doc_name' in st.session_state:
                    del st.session_state['selected_doc_name']
                st.rerun()
            else:
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
                            result = query_document(st.session_state['selected_doc'], summary_query, st.session_state['user_id'])
                            
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
                            result = query_document(st.session_state['selected_doc'], key_points_query, st.session_state['user_id'])
                            
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
                            # Query with user_id for ownership verification
                            result = query_document(st.session_state['selected_doc'], query, st.session_state['user_id'])
                            
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
            1. Tap the **≡** menu icon in the top-left corner to access the sidebar
            2. Upload your PDF in the sidebar that appears
            3. Your document will be processed and ready for questions
            """)

if __name__ == "__main__":
    main()