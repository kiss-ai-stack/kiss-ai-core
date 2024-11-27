from core.agent.agent_stack import AgentStack

agent = AgentStack()
agent.initialize_stack()
print(agent.process_query('what is 1 + 1?'))