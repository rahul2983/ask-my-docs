# Ask My Docs - PDF Q&A Application with Authentication

This application allows you to upload PDF documents and ask questions about their content using Retrieval-Augmented Generation (RAG). The app includes user authentication, enabling multiple users to have their own private document collections.

## Features

- **User Authentication**:
  - User registration and login
  - Password reset functionality
  - User-specific document collections
  - Secure password hashing

- **Document Management**:
  - Upload and manage multiple PDF documents
  - Extract and process text from PDFs
  - Generate embeddings for semantic search using OpenAI or Hugging Face models
  - Store documents and metadata in SQLite database

- **Document Querying**:
  - Query documents using natural language
  - Get answers with source context for verification
  - Mobile-friendly interface with optimized UI for smaller screens
  - Support for both OpenAI and Hugging Face embedding models

## Requirements

- Python 3.8+
- OpenAI API key (recommended)
- Hugging Face account and API token (optional)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/ask-my-docs.git
cd ask-my-docs
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root and add your API keys:
```
# API Keys
OPENAI_API_KEY=your_openai_api_key_here
HUGGINGFACE_API_TOKEN=your_token_here

# Application Settings
DEVICE=cpu
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
EMBEDDING_MODEL=text-embedding-ada-002
USE_OPENAI=true
OPENAI_MODEL=gpt-3.5-turbo

# Database Settings
DB_PATH=pdf_qa.db
VECTOR_STORE_DIR=vector_stores

# Authentication Settings
SECRET_KEY=your-secret-key-for-password-hashing  # Change this to a random string in production
```

## Running the Application

Start the Streamlit server:
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501` in your web browser.

To avoid opening a new browser tab each time, you can use:
```bash
streamlit run app.py --server.headless=true
```
And manually navigate to http://localhost:8501 in your browser.

## How to Use

1. **Register an Account**:
   - Create a new account with a username, email, and password
   - Or log in with your existing credentials

2. **Upload a PDF**: 
   - Use the file uploader in the sidebar to upload a PDF document
   - If a document with the same name already exists, you'll be given options to:
     - Use the existing document
     - Process it as a new document

3. **Select a Document**: 
   - Click on a document in the sidebar to select it for querying
   - The sidebar automatically closes on mobile after selection for better viewing

4. **Ask Questions**: 
   - Type your question in the text input field and press Enter
   - Use quick query buttons for common questions like "Summarize document" or "Key points"

5. **View Answers**: 
   - See the generated answer directly in the main view
   - Explore source context in the expandable section below to verify information

6. **Manage Documents**: 
   - Delete documents you no longer need using the trash icon (🗑️)
   - Reprocess documents with the refresh icon (🔄) if you change embedding models
   - Navigate between documents using the "Choose another document" button
   - Return to the home screen using the "Home" button in the top-right corner

7. **Account Management**:
   - Access account options through the user menu in the top-right corner
   - Log out when you're finished using the application
   - Reset your password if you forget it

## Authentication System

The application includes a complete user authentication system with the following features:

- **User Registration**: Create an account with username, email, and password
- **Secure Password Storage**: Passwords are securely hashed using HMAC-SHA256
- **Login/Logout**: Session-based authentication
- **Password Reset**: Token-based password reset functionality
- **User-Specific Documents**: Each user can only see and access their own documents
- **Access Control**: Documents are associated with specific users for privacy

### Security Notes

- For production use, additional security measures should be implemented:
  - Use HTTPS for all connections
  - Add rate limiting for login attempts
  - Implement proper email-based password reset
  - Consider adding two-factor authentication
  - Use a more secure session management system

## OpenAI Integration

The application supports OpenAI models for both embeddings and language models. This provides significantly better quality answers and more robust embedding.

### Setup OpenAI

1. **Get an OpenAI API Key**:
   - Create an account at [openai.com](https://openai.com)
   - Navigate to [API Keys](https://platform.openai.com/api-keys) and create a new secret key
   - Copy this key to your `.env` file

2. **Configure the .env file**:
   ```
   # API Keys
   OPENAI_API_KEY=your_openai_api_key_here
   
   # Application Settings
   USE_OPENAI=true
   EMBEDDING_MODEL=text-embedding-ada-002
   OPENAI_MODEL=gpt-3.5-turbo  # or gpt-4 for even better responses
   ```

3. **Model Options**:
   - For embeddings: `text-embedding-ada-002` is recommended
   - For the LLM: 
     - `gpt-3.5-turbo` provides good responses at a lower cost
     - `gpt-4` provides the highest quality responses at a higher cost

### Cost Considerations

When using OpenAI models, be aware of the associated costs:
- Embeddings: ~$0.0001 per 1,000 tokens
- GPT-3.5-Turbo: ~$0.002 per 1,000 tokens
- GPT-4: ~$0.06 per 1,000 tokens

For most personal or small-scale use, these costs will be minimal.

## Troubleshooting

### Authentication Issues

- **Can't Login**: Make sure you're using the correct username and password
- **Password Reset**: In this demo version, reset tokens are displayed on screen (in a production environment, these would be sent by email)
- **Database Issues**: If you encounter database errors, check that the DB_PATH in your .env file is correct

### Document Issues

- **Dimension Mismatch Error**: Follow the instructions to reprocess the document
- **GPU/CUDA Issues**: Set `DEVICE=cpu` in your .env file and restart
- **Mobile-Specific Issues**: Ensure you're using the latest version of the app

## Future Enhancements

Some potential improvements to consider:

1. **Enhanced Authentication**:
   - Email verification for new accounts
   - Two-factor authentication
   - OAuth integration (Google, GitHub, etc.)

2. **User Management**:
   - Admin panel for user management
   - User groups and roles
   - Document sharing between users

3. **Document Features**:
   - OCR Support for scanned PDFs
   - Multiple file upload for batch processing
   - Support for additional file types (DOCX, TXT, etc.)

4. **UI Improvements**:
   - Visualization for document statistics
   - Dark mode toggle
   - Progressive Web App support

## How It Works

1. **Authentication Flow**:
   - User credentials stored securely in SQLite
   - Password hashing with HMAC-SHA256
   - Session-based authentication with Streamlit

2. **PDF Processing**: 
   - Extract text from PDF using PyPDFLoader
   - Split text into chunks with RecursiveCharacterTextSplitter
   - Generate embeddings using OpenAI or Sentence Transformers

3. **Storage**:
   - Store document metadata in SQLite with user association
   - Save vector embeddings using FAISS
   - Track embedding type for each document

4. **Query Processing**:
   - Convert query to embedding using the same model that processed the document
   - Retrieve relevant document chunks
   - Generate answer using OpenAI's language models

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.