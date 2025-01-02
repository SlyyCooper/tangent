from tangent import tangent, Agent, run_tangent_loop

client = tangent()

agent = Agent(
    name="Basic Agent",
    model="gpt-4o",
    instructions="You are a simple chatbot that can respond to user requests."
)

run_tangent_loop(agent, stream=True, debug=False)