#from sentence_transformers import SentenceTransformer
import pinecone
import vertexai
import streamlit as st
from vertexai.language_models import TextGenerationModel
from vertexai.language_models import TextEmbeddingModel

#model = SentenceTransformer('all-MiniLM-L6-v2')

model= TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
pinecone.init(api_key='e7fb3d39-83eb-4961-a7ea-6afc42842f45', environment='us-west4-gcp-free')
index = pinecone.Index('langchain-chatbot2')

def find_match(input):
    print("Input at find_match:", input)
   # input_em = model.encode(input).tolist()
    #input_em = encode_texts_to_embeddings(input.split())
    input_em = text_embedding(input)
    print(input_em)
    print("Length:", len(input_em))
    result = index.query(input_em, top_k=2, includeMetadata=True)
    return result['matches'][0]['metadata']['text']+"\n"+result['matches'][1]['metadata']['text']

# Define an embedding method that uses the model
def text_embedding(input) -> list:
    """Text embedding with a Large Language Model."""
    model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
    embeddings = model.get_embeddings([input])
    for embedding in embeddings:
        vector = embedding.values
        print(f"Length of Embedding Vector: {len(vector)}")
    return vector


"""def encode_texts_to_embeddings(input):
    embeddings=model.get_embeddings(input)
    print(embedding.values for embedding in embeddings)
    return [embedding.values for embedding in embeddings]


def encode_texts_to_embeddings(sentences: List[str]) -> List[Optional[List[float]]]:
    try:
        embeddings = model.get_embeddings(sentences)
        return [embedding.values for embedding in embeddings]
    except Exception:
        return [None for _ in range(len(sentences))]
    
# Generator function to yield batches of sentences
def generate_batches(
    sentences: List[str], batch_size: int
) -> Generator[List[str], None, None]:
    for i in range(0, len(sentences), batch_size):
        yield sentences[i : i + batch_size]


def encode_text_to_embedding_batched(
    sentences: List[str], api_calls_per_second: int = 10, batch_size: int = 5
) -> Tuple[List[bool], np.ndarray]:

    embeddings_list: List[List[float]] = []

    # Prepare the batches using a generator
    batches = generate_batches(sentences, batch_size)

    seconds_per_job = 1 / api_calls_per_second

    with ThreadPoolExecutor() as executor:
        futures = []
        for batch in tqdm(
            batches, total=math.ceil(len(sentences) / batch_size), position=0
        ):
            futures.append(
                executor.submit(functools.partial(encode_texts_to_embeddings), batch)
            )
            time.sleep(seconds_per_job)

        for future in futures:
            embeddings_list.extend(future.result())

    is_successful = [
        embedding is not None for sentence, embedding in zip(sentences, embeddings_list)
    ]
    embeddings_list_successful = np.squeeze(
        np.stack([embedding for embedding in embeddings_list if embedding is not None])
    )
    return is_successful, embeddings_list_successful
     
"""    
def query_refiner(conversation, query):
    print("Input at query refiner:", query)

    model = TextGenerationModel.from_pretrained("text-bison@001")
    #should consider refining the prompt further
    response = model.predict(
    prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
    temperature=0.7,
    max_output_tokens=256,
    top_p=0.8,
    top_k=40
    )
    return response.text


def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string