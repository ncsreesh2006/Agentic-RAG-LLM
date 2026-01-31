from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb.config import Settings
import chromadb
from llama_cpp import Llama

def get_token_count(text):
	return len(text)

def manage_history(history, max_history_tokens=1000):
	current_history_str = "\n".join([f"Q: {q}\nA: {a}" for q, a in history])

	while get_token_count(current_history_str) > max_history_tokens and len(history) > 1:
		history.pop(0)
		current_history_str = "\n".join([f"Q: {q}\nA: {a}" for q, a in history])
	
	return history
	
#Actually starting the LLM

llm = Llama(model_path = "llama-2-7b-chat.Q4_K_S.gguf", n_ctx = 4096, verbose=False)

#Processing the text and returning the best match to the LLM

with open("/Users/sreeshchakravarthula/rag-playground/local-llm1/policy.txt", "r") as f:
        raw_text = f.read()

splitter = RecursiveCharacterTextSplitter(chunk_size = 100, chunk_overlap = 20)
chunks = splitter.split_text(raw_text)

client = chromadb.PersistentClient(path = "./my_policy_db1", settings=Settings(allow_reset=True))
client.reset()
collection = client.get_or_create_collection(name = "company_policies")

ids = [f"id_{i}" for i in range(len(chunks))]

metadatas = [{"source": "policy.txt", "page" : 12} for _ in chunks]

collection.add(
        documents = chunks,
        ids=ids,
        metadatas=metadatas
)

print("Enter in a question that can be answered with the information in policy.txt. When you are done with asking questions, type \"DONE\"")

chat_history = []

while True:
	print(">: Enter a question: ")
	question= input()
	if(question.upper() == "DONE"):
		print("GOODBYE")
		break
	else:
#Continuously querying the question and returning the relevant chunk
		history_str = ""
		if(len(chat_history) > 0):
			history_str = "\n".join([f"Human: {q}\nAI: {a}" for q, a in chat_history])
		re_context_prompt = f"""[INST]<<SYS>>Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question. <</SYS>>
		Chat History:
		{history_str}
		Follow up Input: {question}
		Standalone question: [/INST]"""

		re_context_res = llm(re_context_prompt, max_tokens = 100)
		search_query = re_context_res["choices"][0]["text"].strip()
		
		results = collection.query(query_texts=[search_query], n_results=5, include=["documents", "distances", "metadatas"])
		#results = collection.query(query_texts=[question], n_results=5, include=["documents", "distances", "metadatas"])		
		metadata = results['metadatas'][0][0]
		retrieved_distance = results['distances'][0][0]
		if(retrieved_distance > 1):
        		retrieved_code = "NOT IN CONTEXT"
		else:
        		retrieved_code = "\n--\n".join(results['documents'][0])
		
		history_context = "\n".join([f"Q: {q}\nA: {a}" for q, a in chat_history[-3:]])

		

		#chat_history = manage_history(chat_history, max_history_tokens=1000)
	
		#history_context = "\n".join([f"Q: {q}\nA: {a}" for q, a in chat_history])		

		final_prompt = f"""[INST]<<SYS>> You are a helpful assistant that answers questions. Use only the given context and history to answer the question and nothing else. Answer only what is asked in the question and nothing else.
                        <<SYS>>

                        Context:
                        {retrieved_code}

			History:
			{history_context}
		
                        Q: {question}
                        A: [/INST]
                        """
		response = llm(final_prompt, max_tokens = 500)
		print(f"Answer: {response["choices"][0]["text"]}")
	       # print(f"Source: {metadatas[0]['source']}, Page: {metadatas[0]['page']}")
                #print(f"retrieved from {metadata}")
		chat_history.append((question, response))
