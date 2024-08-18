from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os
import warnings
warnings.filterwarnings("ignore")


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text(path):
    text = ""
    with open(path, "rb") as file:
        text = file.read()
    text = text.decode('utf-8')
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=200,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    print("vectorstore called and built: ", vectorstore)
    return vectorstore



def main():
    load_dotenv()
    raw_text = ""
    text_chunks = ""
    vectorstore = None
    raw_text = get_text("./autoRAG.txt")
    text_chunks = get_text_chunks(raw_text)

    vectorstore = get_vectorstore(text_chunks)
    while(1):
        input_text = str(input("Enter input: "))
        print(input_text)
        print(type(input_text))
        if input_text.lower() == "exit":
            break
        retrieved_docs = vectorstore.similarity_search(input_text, k=2)
        print(retrieved_docs)



if __name__ == '__main__':
    main()