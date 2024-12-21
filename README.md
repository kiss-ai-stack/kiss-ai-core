
<div style="text-align: left; margin-bottom: 20px;">
  <img src="https://kiss-ai-stack.github.io/kissaistack.svg" alt="KISS AI Stack Banner" style="max-width: auto; height: 250px">
</div>

# KISS AI Stack - Core

**Effortless AI Stack Building**

Welcome to the core of the **KISS AI Stack**! This module helps you build a stack effortlessly using a simple YAML configuration file. Say goodbye to boilerplate code and embrace minimalism with the **KISS principle** (Keep It Simple, Stupid).

---

## Features

- **Centralized Stack Management**: Manage multiple session-based AI stacks with lifecycle support.
- **Minimal Dependencies**: Built using simple, vanilla vendor libraries.
- **Tool Classification**: Configure tools for your stack to handle specific tasks easily.
- **Supports RAG and Prompt-Based Models**: Choose the model type that suits your needs.
- **Thread-Safe**: Reliable operation in multi-threaded environments.

---

## Installation

Install the core module using pip:

```bash
pip install kiss-ai-stack-core~=0.1.0a28
```

---

## Example Configuration

Here’s an example YAML configuration to set up an AI stack with different tools:

```yaml
stack:
  decision_maker: # Required for tool classification
    name: decision_maker
    role: classify tools for given queries
    kind: prompt  # Choose from 'rag' or 'prompt'
    ai_client:
      provider: openai
      model: gpt-4
      api_key: <your-api-key>

  tools:
    - name: general_queries
      role: process other queries if no suitable tool is found.
      kind: prompt
      ai_client:
        provider: openai
        model: gpt-4
        api_key: <your-api-key>

    - name: document_tool
      role: process documents and provide answers based on them.
      kind: rag  # Retrieval-Augmented Generation
      embeddings: text-embedding-ada-002
      ai_client:
        provider: openai
        model: gpt-4
        api_key: <your-api-key>

  vector_db:
    provider: chroma
    kind: remote # Choose in-memory, storage, or remote options.
    host: 0.0.0.0
    port: 8000
    secure: false
```

---

## Example Python Usage

Use the core module to build and interact with your AI stack:

```python
from kiss_ai_stack import Stacks

async def main():
    try:
        # Initialize a stack in the stack
        await Stacks.bootstrap_stack(stack_id="my_stack", temporary=True)

        # Process a query
        response = await Stacks.generate_answer(stack_id="my_stack", query="What is Retrieval-Augmented Generation?")
        print(response.answer)

    except Exception as ex:
        print(f"An error occurred: {ex}")

# Run the example
import asyncio
asyncio.run(main())
```

---

## How It Works

1. **Stack Initialization**: Use `Stack.bootstrap_stack` to initialize stacks with their configuration and resources.
2. **Query Processing**: Process queries with `Stack.generate_answer`, leveraging tools and AI clients defined in the YAML configuration.
3. **Tool Management**: Define tools to handle specific tasks like document processing or query classification.
4. **Vector Database**: Use the `vector_db` section to define how document embeddings are stored and retrieved for RAG-based tasks. Currently, only `Chroma` is supported.

---

For more information please refer: ![Core documentation](https://kiss-ai-stack.github.io/docs/ai-stack-core/)

## Contributing

We welcome contributions! Submit pull requests or open issues to improve this stack.

---

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
