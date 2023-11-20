from dotenv import load_dotenv
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import pdfplumber


def main():
  load_dotenv()

  # locale.setlocale(locale.LC_ALL, 'zh_CN')
  st.set_page_config(page_title="Ask PDF")
  st.header("Ask PDF💬")
  # 上传文件
  pdf = st.file_uploader("Upload PDF file", type="pdf")

  # 提取文本
  if pdf is not None:
    text = ""
    with pdfplumber.open(pdf) as pdf_reader:
      for page in pdf_reader.pages:
        text += page.extract_text()
    
    # 文本分片
    text_splitter = CharacterTextSplitter(
      separator="\n",
      chunk_size=100,
      chunk_overlap=50,
      length_function=len
    )
    chunks = text_splitter.split_text(text)

    st.write(chunks)
    
    # 创建embeddings
    # embeddings = OpenAIEmbeddings()
    # knowledge_base = FAISS.from_texts(chunks, embeddings)
    # user_question = st.text_input("来向我提问吧：")
    # if user_question:
    #   docs = knowledge_base.similarity_search(user_question)
      
    #   llm = OpenAI()
    #   chain = load_qa_chain(llm, chain_type="stuff")
    #   with get_openai_callback() as cb:
    #     response = chain.run(input_documents=docs, question=user_question)
    #     print(cb)
          
    #   st.write(response)

if __name__ == '__main__':
    main()

