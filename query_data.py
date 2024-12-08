import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate

# from langchain_community.llms.ollama import Ollama
from langchain_openai import ChatOpenAI

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Responda à pergunta com base apenas no seguinte contexto:

{context}

---

Responda à pergunta com base no contexto acima: {question}
"""


def main():
    # Cria a interface de linha de comando (CLI).
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="O texto da consulta.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str):
    # Prepara o banco de dados.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Pesquisa no banco de dados.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt, "\n\n---\n\n")

    model = ChatOpenAI(model="gpt-4o")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Resposta: {response_text.content}\nFontes: {sources}"
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()
