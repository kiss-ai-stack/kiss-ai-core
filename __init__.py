from core.agent.agent_stack import AgentStack

try:
    agent = AgentStack()
    agent.initialize_stack()
    agent.store_documents(['/home/lpsandaruwan/Desktop/kiss-ai-stack/resume.pdf'])
    print(agent.process_query('Give a summary about Lahiru').answer)
except Exception as ex:
    raise ex