import os

from haystack.preview import Pipeline
from haystack.preview.dataclasses.document import Document
from haystack.preview.components.retrievers.memory_bm25_retriever import MemoryBM25Retriever
from haystack.preview.document_stores.memory import MemoryDocumentStore
from haystack.preview.components.generators.openai.gpt35 import GPT35Generator
from haystack.preview.components.builders.prompt_builder import PromptBuilder


docstore = MemoryDocumentStore()
docstore.write_documents(
    [
        Document(text="This is not the answer you are looking for.", metadata={"name": "Obi-Wan Kenobi"}),
        Document(text="This is the way.", metadata={"name": "Mandalorian"}),
        Document(text="The answer to life, the universe and everything is 42.", metadata={"name": "Deep Thought"}),
        Document(text="When you play the game of thrones, you win or you die.", metadata={"name": "Cersei Lannister"}),
        Document(text="Winter is coming.", metadata={"name": "Ned Stark"}),
    ]
)
retriever = MemoryBM25Retriever(document_store=docstore, top_k=3)
# docs = retriever.run(["What is the answer to life, the universe and everything?"])["documents"]

template = """Given the context please answer the question.
Context:
{# We're receiving a list of lists, so we handle it like this #}
{% for doc in documents %}
    {{- doc -}};
{% endfor %}
Question: {{ question }};
Answer:
"""
prompt_builder = PromptBuilder(template)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
generator = GPT35Generator(api_key=OPENAI_API_KEY)

pipe = Pipeline()

pipe.add_component("docs_retriever", retriever)
pipe.add_component("builder", prompt_builder)
pipe.add_component("gpt35", generator)

pipe.connect("docs_retriever.documents", "builder.documents")
pipe.connect("builder.prompt", "gpt35.prompt")


import asyncio


async def main():
    queries = ["What is the answer to life, the universe and everything?", "Which is the way?", "What's coming?"]

    futures = []
    for query in queries:
        data = {"docs_retriever": {"query": query}, "builder": {"question": query}}
        futures.append(pipe.arun(data))

    results = await asyncio.gather(*futures)
    for res in results:
        print(res)
        print()


if __name__ == "__main__":
    asyncio.run(main())
