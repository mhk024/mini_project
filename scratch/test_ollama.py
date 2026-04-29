import time
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="phi")
start = time.time()
print("Generating...")
res = llm.invoke("Write a 100 word summary of Artificial Intelligence.")
end = time.time()
print(f"Response: {res}")
print(f"Time taken: {end - start:.2f}s")
print(f"Words: {len(res.split())}")
