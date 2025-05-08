# Ask My Docs - PDF Q&A Application

This application allows you to upload PDF documents and ask questions about their content using Retrieval-Augmented Generation (RAG). The app extracts text from PDFs, breaks it into chunks, generates embeddings, and uses a language model to answer queries based on the most relevant chunks.

## Features

- Upload and manage multiple PDF documents
- Extract and process text from PDFs
- Generate embeddings for semantic search using OpenAI or Hugging Face models
- Store documents and metadata in SQLite database
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

1. **Upload a PDF**: 
   - Use the file uploader in the sidebar to upload a PDF document
   - If a document with the same name already exists, you'll be given options to:
     - Use the existing document
     - Process it as a new document

2. **Select a Document**: 
   - Click on a document in the sidebar to select it for querying
   - The sidebar automatically closes on mobile after selection for better viewing

3. **Ask Questions**: 
   - Type your question in the text input field and press Enter
   - Use quick query buttons for common questions like "Summarize document" or "Key points"

4. **View Answers**: 
   - See the generated answer directly in the main view
   - Explore source context in the expandable section below to verify information

5. **Manage Documents**: 
   - Delete documents you no longer need using the trash icon (🗑️)
   - Reprocess documents with the refresh icon (🔄) if you change embedding models
   - Navigate between documents using the "Choose another document" button
   - Return to the home screen using the "Home" button in the top-right corner

6. **Mobile Usage**:
   - Tap the **≡** menu icon in the top-left corner to access the sidebar
   - All buttons are larger and full-width for easier tapping on mobile devices

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

## Handling Embedding Model Changes

If you switch between different embedding models (e.g., from Hugging Face to OpenAI), you may encounter a dimension mismatch error when querying previously processed documents:

```
Dimension mismatch error detected. This happens when a document was embedded with a different model.
```

**Why this happens**: Different embedding models produce vectors of different dimensions. For example, OpenAI's embeddings are 1536 dimensions while Hugging Face's model might use 384 dimensions.

**How to fix**:
1. When you see this error, click the 🔄 button next to the document to remove it
2. This will remove the document (but keep the original PDF)
3. Re-upload the document to process it with your current embedding settings

The application tracks which embedding model was used for each document to help you identify when this issue might occur.

## Troubleshooting

### GPU/CUDA Issues

If you encounter an error like `AssertionError: Torch not compiled with CUDA enabled`, follow these steps:

1. Set `DEVICE=cpu` in your `.env` file
2. Restart the application

If you want to use GPU acceleration (which is faster), you'll need to reinstall PyTorch with CUDA support:

```bash
# For CUDA 11.8 (adjust version as needed)
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Deserialization Security Issues

If you encounter this error:
```
ValueError: The de-serialization relies loading a pickle file. Pickle files can be modified to deliver a malicious payload...
```

This is a security feature in newer versions of LangChain. The application has been updated to handle this correctly for both older and newer versions of LangChain.

### Mobile-Specific Issues

- **UI Elements Duplicating**: If UI elements appear multiple times, simply refresh the page (this has been fixed in the latest version)
- **Button Size Too Small**: All buttons have been made full-width for easier tapping on mobile screens

### Other Common Issues

- **Memory Errors**: For large PDFs, reduce `CHUNK_SIZE` or process them in batches
- **Model Downloads**: The first run might take longer as it downloads required models
- **Import Errors**: If you encounter import errors, make sure you've installed the latest versions of langchain and langchain-community
- **File Not Found**: Make sure the vector_stores directory exists in your project folder

## How It Works

1. **PDF Processing**: 
   - Extract text from PDF using PyPDFLoader
   - Split text into chunks with RecursiveCharacterTextSplitter
   - Generate embeddings using OpenAI or Sentence Transformers

2. **Storage**:
   - Store document metadata in SQLite
   - Save vector embeddings using FAISS
   - Track embedding type for each document

3. **Query Processing**:
   - Convert query to embedding using the same model that processed the document
   - Retrieve relevant document chunks
   - Generate answer using OpenAI's language models

## Customization

You can customize various aspects of the application:

- **Embedding Model**: Change `EMBEDDING_MODEL` in the `.env` file to use a different model
- **Language Model**: Change `OPENAI_MODEL` to use a different model (e.g., gpt-4)
- **Chunk Size**: Adjust `CHUNK_SIZE` for different document splitting granularity
- **Retrieval Settings**: Modify the number of chunks retrieved for different query types

## Future Enhancements

Some potential improvements to consider:

1. **User Authentication**: Add login functionality to keep documents private per user
2. **OCR Support**: Implement Optical Character Recognition for scanned PDFs
3. **Multiple File Upload**: Allow batch processing of multiple PDFs
4. **Custom Prompt Templates**: Let users define their own prompts for specific document types
5. **Advanced Search Options**: Add filtering, date range constraints, and metadata search
6. **Visualization**: Add data visualization for document statistics and relationship graphs
7. **Export/Import**: Enable exporting processed documents and importing them on other instances
8. **Progressive Web App**: Convert to PWA for better mobile experience with offline capabilities

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.