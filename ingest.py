from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GooglePalmEmbeddings
from langchain.vectorstores import FAISS
from PyPDF2 import PdfReader

text = ''
vectordb_file_path="palm_index_ipc"
pdf = PdfReader('IPC.pdf')

# read pdf
for page in pdf.pages:
    text += page.extractText()

print('text read completed successfully!')

# divide text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
chunks = text_splitter.split_text(text=text)
print(len(chunks))

# create embeddings
embeddings = GooglePalmEmbeddings(google_api_key='AIzaSyAhdjQL0ziAiWe9ZSFhG2tEn4hGKpis_p4')

# store embeddings
db = FAISS.from_texts(chunks, embedding=embeddings)

db.save_local(vectordb_file_path)

print('embeddings saved successfully!')
