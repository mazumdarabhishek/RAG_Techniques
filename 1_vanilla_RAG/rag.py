from langchain.chains.flare.prompts import PROMPT_TEMPLATE

from myUtils.chroma_db_utils import *
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

llm = ChatGroq(groq_api_key=groq_api_key, model_name='Llama3-8b-8192')
embedder = OllamaEmbeddings(model='mxbai-embed-large')


def run_conversation(llm, retrieved_objects, query):


    context = "\n\n---\n\n".join([doc.page_content for doc, _ in retrieved_objects])
    PROMPT_TEMPLATE = """
    Answer the question based only on the context given below:
    context:
    {context}

    ---
    question: {query}
    """
    prompt = PROMPT_TEMPLATE.format(context=context, query=query)

    result = llm.invoke(prompt)
    print("AI::  ", result.content)
    for doc in retrieved_objects:
        print(f" Source: {doc[0].metadata}\n")



def main(llm, embedder):

    document_dir = input("Path to Document Directory :: ")

    documents = load_documents_from_directory(document_dir)
    chunks = split_into_chunks(documents)
    retriever = load_data_into_chroma(chunks, embedder)

    while True:
        print("Ask a question that I can answer from your documents! type '$bye' to end chat\n")
        query = input("USER:: ")
        if query == "$bye":
            break

        query_embeddings = embedder.embed_query(query)
        retrieved_objects = retriever.similarity_search_by_vector_with_relevance_scores(
            query_embeddings, k=3
        )
        if len(retrieved_objects) == 0 or retrieved_objects[0][1] < 0.7:
            print("Unable to find any relevant document to answer this question")
            continue

        run_conversation(llm, retrieved_objects, query)



    print("\nCHAT ENDED")


if __name__ == "__main__":
    main(llm, embedder)