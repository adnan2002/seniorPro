import boto3
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from dotenv import load_dotenv
import os
import tempfile 

# Load environment variables
load_dotenv()

# Get API keys
openai_api_key = os.getenv('OPENAI_API_KEY')
aws_access_key = os.getenv('aws_access_key')
aws_secret_access_key = os.getenv('aws_secret_access_key')

# Initialize S3 client with credentials
s3 = boto3.client(
    's3',
    region_name='eu-north-1',
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_access_key
)
# S3 bucket and folder information
bucket_name = 'seniorbucket'
folder_name = 'ITCS325-vector-db-openai'

# Create a temporary directory to store downloaded files
with tempfile.TemporaryDirectory() as temp_dir:
    # Download the contents of the S3 folder
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_name)
    for obj in response.get('Contents', []):
        if not obj['Key'].endswith('/'):  # Skip directories
            local_file_path = os.path.join(temp_dir, os.path.basename(obj['Key']))
            s3.download_file(bucket_name, obj['Key'], local_file_path)

    # Initialize OpenAI models
    llm = ChatOpenAI(model="gpt-4", api_key=openai_api_key)
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-large", api_key=openai_api_key)

    # Load the vector store from the temporary directory
    vector_store = FAISS.load_local(temp_dir, embedding_model, allow_dangerous_deserialization=True)

    # Rest of your code remains the same
    retriever = vector_store.as_retriever()

    system_prompt = (
        "Use the given context to answer the question. "
        "If you don't know the answer, say you don't know. "
        "You can be creative"
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, question_answer_chain)

    with open('prompt.txt', 'r') as file:
        query = file.read().strip()

    # Invoke the chain and store the result
    result = chain.invoke({"input": query})

    # Print the result
    print("Question:", query)
    print("Answer:\n", result['answer'])



# def create_vector_db():
#     loader = PyPDFLoader("02-DataRepresentation.pdf")
#     pages = loader.load_and_split()

#     # Convert pages to list of documents
#     documents = [Document(page_content=page.page_content) for page in pages]
#     # Embed the documents
#     embeddings = embedding_model.embed_documents([doc.page_content for doc in documents])

#     # Create a FAISS vector store
#     vector_store = FAISS.from_documents(documents=documents, embedding=embedding_model)

#     # Optionally, save the vector store to disk for later use
#     vector_store.save_local("faiss_vector_store")

#     print("Vector store created and saved successfully.")
