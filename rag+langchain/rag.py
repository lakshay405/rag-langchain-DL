import argparse
import os
import redis
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader

# Configure Redis connection
redis_client = redis.Redis(
    host='',
    port=33641,
    password='',
    ssl=True
)

# Set your OpenAI API key here
OPENAI_API_KEY = ""

CHROMA_PATH = "chroma"
DATA_PATH = "data/books"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Clear Redis cache
    clear_redis_cache()

    # Load data into Redis
    load_data_into_redis()

    # Prepare the DB.
    embedding_function = OpenAIEmbeddings(redis_client=redis_client)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    # Generate response using LangChain's ChatOpenAI model
    model = ChatOpenAI(api_key=OPENAI_API_KEY, redis_client=redis_client)
    response_text = model.predict(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)

# Clear Redis cache
def clear_redis_cache():
    redis_client.flushall()

# Load Markdown data into Redis
def load_data_into_redis():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_redis(chunks)

# Load documents from Markdown files
def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    return documents

# Split text into chunks and process
def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks

# Save chunks to Redis
def save_to_redis(chunks: list[Document]):
    # Save chunks to Redis
    for index, chunk in enumerate(chunks):
        key = f"chunk_{index}"
        redis_client.set(key, chunk.page_content)

if __name__ == "__main__":
    main()
