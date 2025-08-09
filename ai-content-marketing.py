import functools
import io
import operator
import os
import requests
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import HumanMessage, BaseMessage, ToolMessage
from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
from PIL import Image
from bs4 import BeautifulSoup

_ = load_dotenv(find_dotenv())

openai_api_key = os.environ["OPENAI_API_KEY"]
base_url = os.environ["ORCHESTATOR_BASE_URL"]
model_name = os.environ["ORCHESTATOR_MODEL"]


llm = ChatOpenAI(base_url=base_url, model=model_name)

# return_direct=False indica que el resultado no se devuelve directamente al usuario final, sino que regresa al agente para que lo procese o combine con otras respuestas.
@tool("process_search_tool", return_direct=False)
def process_search_tool(url: str):
    """Parse web content with BeautifulSoup"""
    response = requests.get(url=url)
    soup = BeautifulSoup(response.content, "html.parser")
    return soup.get_text()

# max_results=1, la cantidad de busquedas que hace en la web
tavily_tool = TavilySearch(
    max_results=3,
    search_depth="basic"
)
tools = [tavily_tool, process_search_tool]

def create_new_agent(llm: ChatOpenAI, tools: list, system_prompt: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor

def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}

content_marketing_team = ["online_researcher", "blog_manager", "social_media_manager"]

options = ["FINISH"] + content_marketing_team

#
system_prompt = (
    "As a content marketing manager, your role is to oversee the insight between these"
    " workers: {content_marketing_team}. Based on the user's request,"
    " determine which worker should take the next action. Each worker is responsible for"
    " executing a specific task and reporting back thier findings and progress."
    " Once all tasks are completed, indicate 'FINISH'."
)


# Herramienta para estrucutrar las salidas
function_def = {
    "name": "route",
    "description": "Select the next role.",
    "parameters": {
        "title": "routeSchema",
        "type": "object",
        "properties": {
            "next": {
                "type": "string",
                "enum": options
            }
        },
        "required": ["next"]
    }
}

content_marketing_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="messages"),
    ("system",
     "Given the conversation above, who should act next? Or should we FINISH? Select one of: {options}"),
]).partial(options=str(options), content_marketing_team=", ".join(content_marketing_team))

content_marketing_llm_with_tools = llm.bind_tools(
    tools=[function_def],   # herramienta para seleccionar el siguiente rol
    tool_choice="route"     # forzar al modelo a usar esta herramienta
    )

# Conectar el prompt con el modelo como una cadena secuencial
#content_marketing_manager_chain = content_marketing_prompt | content_marketing_llm_with_tools
content_marketing_manager_chain = (content_marketing_prompt | llm.bind_tools(
    tools=[function_def], 
    tool_choice="route"
    ))

online_researcher_agent = create_new_agent(
    llm,
    tools,
    """Your primary role is to function as an intelligent online research assistant, adept at scouring 
    the internet for the latest and most relevant trending stories across various sectors like politics, technology, 
    health, culture, and global events. You possess the capability to access a wide range of online news sources, 
    blogs, and social media platforms to gather real-time information."""
    )

def online_researcher_node(state):
    return agent_node(state, agent=online_researcher_agent, name="online_researcher")

blog_manager_agent = create_new_agent(
    llm, tools,
    """You are a Blog Manager. The role of a Blog Manager encompasses several critical responsibilities aimed at transforming initial drafts into polished, SEO-optimized blog articles that engage and grow an audience. Starting with drafts provided by online researchers, the Blog Manager must thoroughly understand the content, ensuring it aligns with the blog's tone, target audience, and thematic goals. Key responsibilities include:

    1. Content Enhancement: Elevate the draft's quality by improving clarity, flow, and engagement. This involves refining the narrative, adding compelling headers, and ensuring the article is reader-friendly and informative.

    2. SEO Optimization: Implement best practices for search engine optimization. This includes keyword research and integration, optimizing meta descriptions, and ensuring URL structures and heading tags enhance visibility in search engine results.

    3. Compliance and Best Practices: Ensure the content adheres to legal and ethical standards, including copyright laws and truth in advertising. The Blog Manager must also keep up with evolving SEO strategies and blogging trends to maintain and enhance content effectiveness.

    4. Editorial Oversight: Work closely with writers and contributors to maintain a consistent voice and quality across all blog posts. This may also involve managing a content calendar, scheduling posts for optimal engagement, and coordinating with marketing teams to support promotional activities.

    5. Analytics and Feedback Integration: Regularly review performance metrics to understand audience engagement and preferences. Use this data to refine future content and optimize overall blog strategy.

    In summary, the Blog Manager plays a pivotal role in bridging initial research and the final publication by enhancing content quality, ensuring SEO compatibility, and aligning with the strategic objectives of the blog. This position requires a blend of creative, technical, and analytical skills to successfully manage and grow the blog's presence online.""")



def blog_manager_node(state):
    return agent_node(state, agent=blog_manager_agent, name="blog_manager")


social_media_manager_agent = create_new_agent(
    llm, tools,
    """You are a Social Media Manager. The role of a Social Media Manager, particularly for managing Twitter content, involves transforming research drafts into concise, engaging tweets that resonate with the audience and adhere to platform best practices. Upon receiving a draft from an online researcher, the Social Media Manager is tasked with several critical functions:

    1. Content Condensation: Distill the core message of the draft into a tweet, which typically allows for only 280 characters. This requires a sharp focus on brevity while maintaining the essence and impact of the message.

    2. Engagement Optimization: Craft tweets to maximize engagement. This includes the strategic use of compelling language, relevant hashtags, and timely topics that resonate with the target audience.

    3. Compliance and Best Practices: Ensure that the tweets follow Twitter’s guidelines and best practices, including the appropriate use of mentions, hashtags, and links. Also, adhere to ethical standards, avoiding misinformation and respecting copyright norms.
    
    4. Your output have been a tweet (<=280 chars)

    In summary, the Social Media Manager's role is crucial in leveraging Twitter to disseminate information effectively, engage with followers, and build the brand’s presence online. This position combines creative communication skills with strategic planning and analysis to optimize social media impact.""")


def social_media_manager_node(state):
    return agent_node(state, agent=social_media_manager_agent, name="social_media_manager")

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str
    
workflow = StateGraph(AgentState)

def content_marketing_manager_node(state):
    result = content_marketing_manager_chain.invoke(state)
    # Extraer el argumento 'next' desde la llamada a herramienta
    if not result.tool_calls:
        raise ValueError("No tool_call found in response.")
    tool_call = result.tool_calls[0]
    tool_result = tool_call["args"]["next"]

    tool_message = ToolMessage(
        tool_call_id=tool_call["id"],
        content=tool_result  # puede ser {"next": "..."} o el nombre directamente
    )

    return {
        "messages": state["messages"] + [result, tool_message],
        "next": tool_result
    }


# Definición de los nodos
workflow.add_node("content_marketing_manager", content_marketing_manager_node)
#workflow.add_node("content_marketing_manager", content_marketing_manager_chain)
workflow.add_node("online_researcher", online_researcher_node)
workflow.add_node("blog_manager", blog_manager_node)
workflow.add_node("social_media_manager", social_media_manager_node)

# Transiciones
workflow.add_edge("online_researcher", "content_marketing_manager")
workflow.add_edge("blog_manager", "content_marketing_manager")
workflow.add_edge("social_media_manager", "content_marketing_manager")

# Rutas conicionales
conditional_map = {
    "online_researcher": "online_researcher",
    "blog_manager": "blog_manager",
    "social_media_manager": "social_media_manager",
    "FINISH": END
}

def select_next_step(state):
    return state["next"]

# Anhadiendo las transiciones condicionales
workflow.add_conditional_edges(
    "content_marketing_manager", select_next_step, conditional_map)

# Establecer punto de entrada
workflow.set_entry_point("content_marketing_manager")

multiagent = workflow.compile()


AGENT_NODES = {"online_researcher", "blog_manager", "social_media_manager"}

step_count = 0
for event in multiagent.stream(
    {
        "messages": [
            HumanMessage(
                content=(
                    """Write me a report on Nanomaterials in Health. After the research on Nanomaterials in Health, pass the findings to the blog manager to generate the final blog article. Once done, pass it to the social media manager to write a tweet on the subject."""
                )
            )
        ],
    },
    {"recursion_limit": 150}
):
    if "__end__" in event:
        continue

    step_count += 1
    print(f"🔄 Paso {step_count}:")

    # Recorremos los nodos que emitieron algo en este evento
    for node_name, payload in event.items():
        if node_name not in AGENT_NODES:
            continue  # ignoramos el manager y otros nodos

        msgs = payload.get("messages", [])
        if not msgs:
            continue

        last_msg = msgs[-1]
        if isinstance(last_msg, HumanMessage):
            print(f"🧠 {node_name} dice:\n{last_msg.content}\n")

    print("-" * 50)

print(f"✅ Flujo completo. Total de pasos ejecutados: {step_count}")

# ===================== Create a image graph =====================
graph_image = multiagent.get_graph(xray=True)
png_bytes = graph_image.draw_mermaid_png()
image = Image.open(io.BytesIO(png_bytes))
image.save("ai_analyst_generator.png")
print("✅ Graph image saved as graph.png")