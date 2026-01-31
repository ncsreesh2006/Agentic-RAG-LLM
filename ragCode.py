from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, util

#Just for splitting the document into chunks
with open ("policy.txt", "r") as ragDoc:
	raw_text = ragDoc.read()
	

text_splitter = RecursiveCharacterTextSplitter(
	chunk_size = 100,
	chunk_overlap = 20
)

chunks = text_splitter.split_text(raw_text)

#For actually encoding the chunks into vectors/embeddings

embedder = SentenceTransformer('all-MiniLM-L6-v2')

chunk_embeddings = embedder.encode(chunks)

question = "Is travel covered for airplanes?"
question_embedding = embedder.encode(question)

hits = util.semantic_search(question_embedding, chunk_embeddings, top_k=1)
best_chunk_idx = hits[0][0]['corpus_id']
best_score = hits[0][0]['score']

retrieved_context = chunks[best_chunk_idx]
print(f"Retrieved with score {best_score:.4f}: {retrieved_context}")
