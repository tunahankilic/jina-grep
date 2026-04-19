from langchain.agents import create_agent
from langchain_community.tools import ShellTool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from skill_registry import Skill, SkillRegistry


def route_skill(llm: ChatOllama, skills: list[Skill], query: str) -> str:
    skill_list = "\n".join(f"- {s.name}: {s.description}" for s in skills)
    prompt = f"""Select the most appropriate skill for this user query.

Available skills:
{skill_list}

User query: {query}

Reply with ONLY the skill name, nothing else."""
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


def run_agent(llm: ChatOllama, skill: Skill, query: str) -> str:
    agent = create_agent(llm, tools=[ShellTool()], system_prompt=skill.content)
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content


def main() -> None:
    registry = SkillRegistry("skills")
    llm = ChatOllama(model="gemma4:e4b-nvfp4")

    print("Dynamic Skill Agent (Ctrl+C to exit)\n")

    while True:
        try:
            query = input("Query: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not query:
            continue

        skill_name = route_skill(llm, registry.list(), query)
        skill = registry.get(skill_name)

        if skill is None:
            skill = registry.list()[0]
            print(f"[Router returned unknown skill '{skill_name}', falling back to '{skill.name}']")
        else:
            print(f"[Skill: {skill.name}]")

        answer = run_agent(llm, skill, query)
        print(f"\n{answer}\n")


if __name__ == "__main__":
    main()
