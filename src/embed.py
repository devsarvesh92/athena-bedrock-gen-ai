import boto3
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.document_loaders import JSONLoader
from langchain.vectorstores import faiss

## Bedrock client
bedrock_client = boto3.client(service_name="bedrock-runtime")
bedrock_embedings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1", client=bedrock_client
)


#### Store emdedings
def load_schema():
    return JSONLoader(
        file_path="./data/data_catalog.json",
        text_content=False,
        jq_schema=".",
    ).load()


def generate_vector_embedings():
    vector_store_faiss = faiss.FAISS.from_documents(
        documents=load_schema(), embedding=bedrock_embedings
    )
    vector_store_faiss.save_local("faiss_index")


if __name__ == "__main__":
    generate_vector_embedings()
