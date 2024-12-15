<div style="text-align: center; margin-bottom: 20px;">
  <img src="https://kiss-ai-stack.github.io/kissaistack.svg" alt="KISS AI Stack Banner" style="max-width: auto; height: 20%;">
</div>

# KISS AI Stack - Core

**Effortless AI Agent Building**

Welcome to the core of the **KISS AI Stack**! This module helps you build an AI agent effortlessly using a simple YAML configuration file. Say goodbye to boilerplate code and embrace minimalism with the **KISS principle** (Keep It Simple, Stupid).

---

## Features

- **Centralized Agent Management**: Manage multiple session-based AI agents with lifecycle support.
- **Minimal Dependencies**: Built using simple, vanilla vendor libraries.
- **Tool Classification**: Configure tools for your agent to handle specific tasks easily.
- **Supports RAG and Prompt-Based Models**: Choose the model type that suits your needs.
- **Thread-Safe**: Reliable operation in multi-threaded environments.

---

## Installation

Install the core module using pip:

```bash
pip install kiss-ai-stack-core
```

---

## Example Configuration

Here’s an example YAML configuration to set up an AI agent with different tools:

```yaml
agent:
  classifier: # Required for tool classification
    name: classifier
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
    kind: remote # Choose in-memory, storage or remote options.
    host: 0.0.0.0
    port: 8000
    secure: false
```

---

## Example Python Usage

Use the core module to build and interact with your AI agent:

```python
from kiss_ai_stack import AgentStack

async def main():
    try:
        # Initialize an agent in the stack
        await AgentStack.bootstrap_agent(agent_id="my_agent", temporary=True)

        # Process a query
        response = await AgentStack.generate_answer(agent_id="my_agent", query="What is KISS AI Stack?")
        print(response.answer)

    except Exception as ex:
        print(f"An error occurred: {ex}")

# Run the example
import asyncio
asyncio.run(main())
```

---

## How It Works

1. **Agent Initialization**: Use `AgentStack.bootstrap_agent` to initialize agents with their configuration and resources.
2. **Query Processing**: Process queries with `AgentStack.generate_answer`, leveraging tools and AI clients defined in the YAML configuration.
3. **Tool Management**: Define tools to handle specific tasks like document processing or query classification.
4. **Vector Database**: Use the `vector_db` section to define how document embeddings are stored and retrieved for RAG-based tasks. Currently, `Chroma` is supported.

---

## Documentation

### Key Methods

- `bootstrap_agent(agent_id: str, temporary: bool)`: Initialize a new agent session.
- `generate_answer(agent_id: str, query: Union[str, Dict, List])`: Process a query and return a response.

### Configuration Highlights

- **AI Client**: Configure the provider, model, and API key for supported services like OpenAI.
- **Tools**: Define tools such as general-purpose query handlers or document processors.
- **Vector Database**: Set up in-memory or persistent storage for RAG-based tasks.

---

## Contributing

We welcome contributions! Submit pull requests or open issues to improve this stack.

---

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

