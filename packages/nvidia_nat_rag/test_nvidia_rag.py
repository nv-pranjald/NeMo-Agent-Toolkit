#!/usr/bin/env python3
"""Standalone test script for nvidia_rag library."""

import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MILVUS_DB = "/home/pranjald/Code/github/NeMo-Agent-Toolkit/packages/nvidia_nat_rag/milvus-lite.db"
COLLECTION_NAME = "test_collection"


async def test_search():
    """Test nvidia_rag search function."""
    from nvidia_rag import NvidiaRAG
    from nvidia_rag.utils.configuration import NvidiaRAGConfig

    print("\n" + "=" * 60)
    print("Testing nvidia_rag.search()")
    print("=" * 60)

    # Initialize
    config = NvidiaRAGConfig()
    config.vector_store.url = MILVUS_DB
    
    print(f"Vector Store URL: {config.vector_store.url}")
    print(f"Embedding Model: {config.embeddings.model_name}")
    print(f"Reranker Model: {config.ranking.model_name}")

    rag = NvidiaRAG(config=config)
    print("NvidiaRAG initialized successfully")

    # Test search
    query = "poem"
    print(f"\nSearching for: '{query}'")
    print("-" * 40)

    try:
        citations = await asyncio.wait_for(
            rag.search(
                query=query,
                collection_names=[COLLECTION_NAME],
                vdb_endpoint=MILVUS_DB,
                reranker_top_k=3,
                vdb_top_k=10,
            ),
            timeout=60.0
        )
        
        print(f"Search returned: {type(citations)}")
        if citations and hasattr(citations, 'results'):
            print(f"Number of results: {len(citations.results)}")
            for i, result in enumerate(citations.results[:3]):
                print(f"\nResult {i+1}:")
                print(f"  Document: {getattr(result, 'document_name', 'N/A')}")
                content = getattr(result, 'content', '')[:200] if hasattr(result, 'content') else 'N/A'
                print(f"  Content: {content}...")
        else:
            print("No results found")
            
    except asyncio.TimeoutError:
        print("ERROR: Search timed out after 60 seconds")
    except Exception as e:
        print(f"ERROR: {e}")


async def test_generate():
    """Test nvidia_rag generate function."""
    from nvidia_rag import NvidiaRAG
    from nvidia_rag.utils.configuration import NvidiaRAGConfig

    print("\n" + "=" * 60)
    print("Testing nvidia_rag.generate()")
    print("=" * 60)

    # Initialize
    config = NvidiaRAGConfig()
    config.vector_store.url = MILVUS_DB

    print(f"LLM Model: {config.llm.model_name}")

    rag = NvidiaRAG(config=config)
    print("NvidiaRAG initialized successfully")

    # Test generate
    query = "What is the poem about?"
    print(f"\nQuery: '{query}'")
    print("-" * 40)

    try:
        response = await asyncio.wait_for(
            rag.generate(
                messages=[{"role": "user", "content": query}],
                use_knowledge_base=True,
                collection_names=[COLLECTION_NAME],
                vdb_endpoint=MILVUS_DB,
            ),
            timeout=60.0
        )

        print(f"Response type: {type(response)}")
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            print("\nStreaming response:")
            full_text = []
            async for chunk in response.generator:
                if chunk.startswith("data: "):
                    chunk = chunk[6:].strip()
                if chunk and chunk != "[DONE]":
                    try:
                        import json
                        data = json.loads(chunk)
                        if "choices" in data:
                            delta = data["choices"][0].get("delta", {})
                            text = delta.get("content", "")
                            if text:
                                full_text.append(text)
                                print(text, end="", flush=True)
                    except:
                        pass
            print("\n\nFull response:", "".join(full_text))

    except asyncio.TimeoutError:
        print("ERROR: Generate timed out after 60 seconds")
    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    print("NVIDIA RAG Standalone Test")
    print("=" * 60)
    
    # Test search first
    asyncio.run(test_search())
    
    # Then test generate
    # asyncio.run(test_generate())
