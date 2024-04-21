import boto3

# Vector embedings
# Titan-v2
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock


# vector store
from langchain.vectorstores import faiss

## LLM
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


## Bedrock client
bedrock_client = boto3.client(service_name="bedrock-runtime")
bedrock_embedings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1", client=bedrock_client
)

def get_llm(model_id:str):
    llm = Bedrock(
        model_id=model_id,
        client=bedrock_client,
        model_kwargs={"maxTokens": 400},
    )
    return llm


prompt_template = """
It is important that the SQL query complies with Athena syntax. \
Always generate presto queries
Always add databasename.tablename in the query
Convert report_date alway to date before doing comparison
<context>
{context}
<context>

Question: {question}
Assistant: """

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


def get_query(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )
    answer = qa({"query": query})
    return answer['result']


if __name__ == "__main__":
    faiss_index = faiss.FAISS.load_local("faiss_index", bedrock_embedings,allow_dangerous_deserialization=True)
    llm=get_llm(model_id="ai21.j2-ultra-v1")
    while True:
        user_input = input("Ask questions (or 'q' to quit): ")
        if user_input.lower() == 'q':
            break  # Exit the loop if the user enters 'q'
        else:
            # Process the user input
            # "get payment schedules after april 2024 for tenant perpay"
            print(get_query(llm=llm,vectorstore_faiss=faiss_index,query=user_input))
        

