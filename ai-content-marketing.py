import functools
import operator
import os
import requests
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import HumanMessage, BaseMessage
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
from bs4 import BeautifulSoup

_ = load_dotenv(find_dotenv())

openai_api_key = os.environ["OPENAI_API_KEY"]
base_url = os.environ["ORCHESTATOR_BASE_URL"]
model_name = os.environ["ORCHESTATOR_MODEL"]


llm = ChatOpenAI(base_url=base_url, model=model_name)


# prompt = "Dame 3 ideas de títulos para un blog sobre inteligencia artificial en marketing digital y generador de contenido."
# response = llm.invoke(prompt)
# print(response.content)


@tool("process_search_tool", return_direct=False)
def process_search_tool(url: str):
    """Parse web content with BeautifulSoup"""
    response = requests.get(url=url)
    soup = BeautifulSoup(response.content, "html.parser")
    return soup.get_text()

# max_results=1, la cantidad de busquedas que hace en la web
tools = [TavilySearchResults(max_results=1), process_search_tool]

def create_new_agent(llm: ChatOpenAI,
                  tools: list,
                  system_prompt: str):
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

#
system_prompt = (
    "Como gerente de marketing de contenidos, tu rol es supervisar la interacción entre estos" "trabajadores: {content_marketing_team}. Según la solicitud del" "usuario, determina qué trabajador debe realizar la siguiente acción. Cada trabajador es" "responsable de ejecutar una tarea específica y de informar sus hallazgos y avances. Una vez" "que todas las tareas estén completadas, indica 'FINISH'."
)

options = ["FINISH"] + content_marketing_team

function_def = {
    "name": "route",
    "description": "Select the next role.",
    "parameters": {
        "title": "routeSchema",
        "type": "object",
        "properties": {"next": {"title": "Next", "anyOf": [{"enum": options}]}},
        "required": ["next"]
    }
}

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="messages"),
    ("system",
     "Given the conversation above, who should act next? Or should we FINISH? Select one of: {options}"),
]).partial(options=str(options), content_marketing_team=", ".join(content_marketing_team))

content_marketing_manager_chain = (prompt | llm.bind_functions(
    functions=[function_def], function_call="route") | JsonOutputFunctionsParser())

online_researcher_agent = create_new_agent(
    llm,
    tools,
    """Tu función principal es actuar como un asistente de investigación en línea inteligente, experto en buscar en internet las historias más recientes y relevantes sobre diversos sectores como política, tecnología, salud, cultura y eventos globales. Tienes la capacidad de acceder a una amplia gama de fuentes de noticias en línea, blogs y plataformas de redes sociales para recopilar información en tiempo real.
    """
    )

online_researcher_node = functools.partial(
    agent_node, agent=online_researcher_agent, name="online_researcher"
)

blog_manager_agent = create_new_agent(
    llm, tools,
    """Eres un Gestor de Blog responsable de transformar borradores en artículos pulidos, optimizados para SEO y alineados con la voz del blog y su audiencia. Tus tareas clave incluyen mejorar la claridad y estructura del contenido, aplicar estrategias SEO, asegurar cumplimiento legal, mantener la coherencia editorial, y usar métricas para mejorar el rendimiento del blog.
    """)


blog_manager_node = functools.partial(
    agent_node, agent=blog_manager_agent, name="blog_manager")


social_media_manager_agent = create_new_agent(
    llm, tools,
    """Eres un Gestor de Redes Sociales especializado en convertir borradores en tweets concisos y atractivos. Tu labor incluye condensar el mensaje principal en 280 caracteres, optimizar la interacción con lenguaje persuasivo y hashtags relevantes, y cumplir con las buenas prácticas y normativas de Twitter para maximizar el impacto de la marca.
    """)

social_media_manager_node = functools.partial(
    agent_node, agent=social_media_manager_agent, name="social_media_manager")

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str
    
workflow = StateGraph(AgentState)

print(">>>> content_marketing_manager_chain:", id(content_marketing_manager_chain))
workflow.add_node(key="content_marketing_manager", node=content_marketing_manager_chain)

print(">>>> online_researcher_node:", id(online_researcher_node))
workflow.add_node(key="online_researcher", node=online_researcher_node)

print(">>>> blog_manager_node:", id(blog_manager_node))
workflow.add_node(key="blog_manager", node=blog_manager_node)

print(">>>> social_media_manager_node:", id(social_media_manager_node))
workflow.add_node(key="social_media_manager", node=social_media_manager_node)

print("---------- FLAG -------------")

for member in content_marketing_team:
    workflow.add_edge(start_key=member, end_key="content_marketing_manager")

conditional_map = {k: k for k in content_marketing_team}

conditional_map['FINISH'] = END

workflow.add_conditional_edges(
    "content_marketing_manager", lambda x: x["next"], conditional_map)

workflow.set_entry_point("content_marketing_manager")

multiagent = workflow.compile()

for s in multiagent.stream(
    {
        "messages": [
            HumanMessage(
                content="""Escríbeme un informe sobre el Comportamiento Agéntico. Después de la investigación, pasa los hallazgos al gestor del blog para que genere el artículo final. Una vez hecho, pásalo al gestor de redes sociales para que redacte un tweet sobre el tema.
                """
            )
        ],
    },
    # Maximum number of steps to take in the multiagent
    {"recursion_limit": 150}
):
    if not "__end__" in s:
        print(s, end="\n\n-----------------\n\n")