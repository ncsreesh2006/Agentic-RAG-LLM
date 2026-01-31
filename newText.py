from llama_cpp import Llama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, util

print(">: Enter a question: ")
question = input()

with open("policy.txt", "r") as f:
	read_file = f.read()

transformer = SentenceTransformer("all-MiniLM-L6-v2")
splitter = RecursiveCharacterTextSplitter(chunk_size = 100, chunk_overlap = 20)
chunks = splitter.split_text(read_file)
chunk_embedding = transformer.encode(chunks)

question_embedding = transformer.encode(question)

top_result = util.semantic_search(question_embedding, chunk_embedding, top_k = 1)

llm = Llama(model_path = "llama-2-7b-chat.Q4_K_S.gguf", n_ctx = 4096)

prompt = f"""[INST]<<SYS>> You are a chatbot that answers questions according to a given context. Use only the given context and nothing else to answer the questions.
<</SYS>>

Context:
{top_result}

Q: {question}
A: [/INST]
"""

response = llm(prompt, max_tokens = 2048)

print(response["choices"][0])
print(top_result)
