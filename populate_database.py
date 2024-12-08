import argparse
import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma
from dotenv import load_dotenv
import openai

# Carrega as variáveis de ambiente. Assume que o projeto contém um arquivo .env com as chaves da API.
load_dotenv()
# ---- Define a chave da API da OpenAI
# Altere o nome da variável de ambiente de "OPENAI_API_KEY" para o nome especificado no
# seu arquivo .env.
openai.api_key = os.environ["OPENAI_API_KEY"]

CHROMA_PATH = "chroma"
DATA_PATH = "data"


def main():
    # Verifica se o banco de dados deve ser limpo (usando o parâmetro --reset).
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reset", action="store_true", help="Reinicia o banco de dados."
    )
    args = parser.parse_args()
    if args.reset:
        print("✨ Limpando o banco de dados")
        clear_database()

    # Cria (ou atualiza) o banco de dados.
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)


def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document]):
    # Carrega o banco de dados existente.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Calcula os IDs das páginas.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Adiciona ou atualiza os documentos.
    existing_items = db.get(include=[])  # IDs são sempre incluídos por padrão
    existing_ids = set(existing_items["ids"])
    print(f"Número de documentos existentes no banco de dados: {len(existing_ids)}")

    # Adiciona apenas documentos que não existem no banco de dados.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adicionando novos documentos: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("✅ Nenhum novo documento para adicionar")


def calculate_chunk_ids(chunks):
    # Isso criará IDs como "data/monopoly.pdf:6:2"
    # Fonte da Página : Número da Página : Índice do Fragmento

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # Se o ID da página for o mesmo que o último, incrementa o índice.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calcula o ID do fragmento.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Adiciona o ID aos metadados do fragmento.
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()
