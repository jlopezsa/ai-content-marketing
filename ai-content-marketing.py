import functools
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
from bs4 import BeautifulSoup

_ = load_dotenv(find_dotenv())

openai_api_key = os.environ["OPENAI_API_KEY"]
base_url = os.environ["ORCHESTATOR_BASE_URL"]
model_name = os.environ["ORCHESTATOR_MODEL"]


llm = ChatOpenAI(base_url=base_url, model=model_name)


@tool("process_search_tool", return_direct=False)
def process_search_tool(url: str):
    """Parse web content with BeautifulSoup"""
    response = requests.get(url=url)
    soup = BeautifulSoup(response.content, "html.parser")
    return soup.get_text()

# max_results=1, la cantidad de busquedas que hace en la web
tavily_tool = TavilySearch(
    max_results=1,
    search_depth="basic"
)
tools = [tavily_tool, process_search_tool]

def create_new_agent(llm: ChatOpenAI, tools: list, system_prompt: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"), # MessagesPlaceholder: This substitutes a variable with a list of messages provided in real time
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor

def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}

content_marketing_team = ["online_researcher", "blog_manager", "social_media_manager"]

#
system_prompt = (
    "Como gerente de marketing de contenidos, tu rol es supervisar la interacción entre estos" "trabajadores: {content_marketing_team}. Según la solicitud del" "usuario, determina qué trabajador debe realizar la siguiente acción. Cada trabajador es" "responsable de ejecutar una tarea específica y de informar sus hallazgos y avances. Una vez" "que todas las tareas estén completadas, indica 'FINISH'."
)

options = ["FINISH"] + content_marketing_team

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
content_marketing_manager_chain = content_marketing_prompt | content_marketing_llm_with_tools

online_researcher_agent = create_new_agent(
    llm,
    tools,
    """Tu función principal es actuar como un asistente de investigación en línea inteligente, experto en buscar en internet las historias más recientes y relevantes sobre diversos sectores como política, tecnología, salud, cultura y eventos globales. Tienes la capacidad de acceder a una amplia gama de fuentes de noticias en línea, blogs y plataformas de redes sociales para recopilar información en tiempo real.
    """
    )

def online_researcher_node(state):
    return agent_node(state, agent=online_researcher_agent, name="online_researcher")

blog_manager_agent = create_new_agent(
    llm, tools,
    """Eres un Gestor de Blog responsable de transformar borradores en artículos pulidos, optimizados para SEO y alineados con la voz del blog y su audiencia. Tus tareas clave incluyen mejorar la claridad y estructura del contenido, aplicar estrategias SEO, asegurar cumplimiento legal, mantener la coherencia editorial, y usar métricas para mejorar el rendimiento del blog.
    """)


def blog_manager_node(state):
    return agent_node(state, agent=blog_manager_agent, name="blog_manager")


social_media_manager_agent = create_new_agent(
    llm, tools,
    """Eres un Gestor de Redes Sociales especializado en convertir borradores en tweets concisos y atractivos. Tu labor incluye condensar el mensaje principal en 280 caracteres, optimizar la interacción con lenguaje persuasivo y hashtags relevantes, y cumplir con las buenas prácticas y normativas de Twitter para maximizar el impacto de la marca.
    """)

def social_media_manager_node(state):
    return agent_node(state, agent=social_media_manager_agent, name="social_media_manager")

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str
    
workflow = StateGraph(AgentState)

def content_marketing_manager_node(state):
    result = content_marketing_manager_chain.invoke(state)
    # Extraer el argumento 'next' desde la llamada a herramienta
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


for s in multiagent.stream(
    {
        "messages": [
            HumanMessage(
                content="""Escríbeme un informe sobre frameworks para implementar Agentes de IA. Después de la investigación, pasa los hallazgos al gestor del blog para que genere el artículo final. Una vez hecho, pásalo al gestor de redes sociales para que redacte un tweet sobre el tema.
                """
            )
        ],
    },
    # Maximum number of steps to take in the multiagent
    {"recursion_limit": 150}
):
    if not "__end__" in s:
        print(s, end="\n\n-----------------\n\n")