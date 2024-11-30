
# kiss-ai-stack - Core

**KISS AI Stack's AI Agent Builder**

Welcome to the core of the **KISS AI Stack**! This module helps you build AI agents effortlessly, using a simple YAML configuration file. With this core, you don't need to worry about boilerplate code. It is designed to keep things minimal and efficient, following the **KISS principle** (Keep It Simple, Stupid).

### Features:
- **Build AI agents quickly**: Just provide a YAML file to configure the agent.
- **Minimal dependencies**: Built using simple, vanilla vendor libraries.
- **Tool classification**: Easily configure tools for your agent to handle specific tasks.
- **Supports RAG and prompt-based models**: Choose the model type that suits your needs.

### Installation

To install the core module, run:

```bash
pip install kiss-ai-stack-core
```

### Example Configuration

Below is an example YAML configuration for setting up an AI agent with different tools:

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
      role: Process documents and provide answers based on them.
      kind: rag  # Retrieval-Augmented Generation
      embeddings: text-embedding-ada-002
      ai_client:
        provider: openai
        model: gpt-4
        api_key: <your-api-key>

  vector_db:
    provider: chroma
    kind: in_memory  # Choose in-memory or persistent storage options.
```

### Example Python Usage

Once the core is installed, you can use it to build and interact with your AI agent as shown in the example below:

```python
from kiss_ai_stack.core.agent import AgentStack

try:
    # Initialize the AI agent stack
    agent = AgentStack()
    agent.initialize_stack()  # Set up the agent using the provided configuration
    
    # Store documents for the agent to process
    agent.store_documents(['./yourfile.pdf'])
    
    # Process a query and get the answer
    print(agent.process_query('Give a summary about yourfile contents.').answer)
    
except Exception as ex:
    print(f"An error occurred: {ex}")
    raise ex
```

### How It Works:
1. **Agent Setup**: The agent is initialized with the provided configuration (defined in the YAML file). The configuration specifies which AI clients and tools to use.
2. **Tools**: Each tool is defined by its type (`prompt` or `rag`) and is linked to an AI client, such as OpenAI’s GPT-4. Tools handle different tasks like document processing or classifying queries.
3. **Vector DB**: The `vector_db` section allows you to configure the database for storing and retrieving document embeddings. Currently, `Chroma` is supported as an in-memory solution.

### Documentation:
- **AI Client**: You can configure the AI client with the provider, model, and API key for any supported service like OpenAI.
- **Tools**: Define the tools that the agent will use, such as a general-purpose query tool or a document processing tool.
- **Vector Database**: The `vector_db` section defines how the agent stores document embeddings and retrieves them for RAG-based tasks.

### Contributing

We welcome contributions! If you'd like to improve this stack, feel free to submit pull requests or open issues for discussion.

### License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
