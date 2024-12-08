# from langchain_community.embeddings.ollama import OllamaEmbeddings
# from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import openai
import os

# Carrega as variáveis de ambiente. Assume que o projeto contém um arquivo .env com as chaves da API.
load_dotenv()
# ---- Define a chave da API da OpenAI
# Altere o nome da variável de ambiente de "OPENAI_API_KEY" para o nome especificado no
# seu arquivo .env.
openai.api_key = os.environ["OPENAI_API_KEY"]


def get_embedding_function():
    # embeddings = BedrockEmbeddings(
    #     credentials_profile_name="default", region_name="us-east-1"
    # )
    # embeddings = OllamaEmbeddings(model="nomic-embed-text")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    return embeddings
