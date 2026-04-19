import io
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="langchain_community.tools.shell")

from langchain.agents import create_agent
from langchain_community.tools import ShellTool


class QuietShellTool(ShellTool):
    def _run(self, commands: str | list[str], **kwargs) -> str:
        stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            return super()._run(commands, **kwargs)
        finally:
            sys.stdout = stdout
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from rich.console import Console
from rich.rule import Rule
from rich.theme import Theme

from skill_registry import Skill, SkillRegistry

theme = Theme({
    "routing": "dim cyan",
    "skill": "bold green",
    "step": "bold yellow",
    "cmd": "cyan",
    "output": "dim white",
    "answer": "white",
})
console = Console(theme=theme)


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
    agent = create_agent(llm, tools=[QuietShellTool()], system_prompt=skill.content)

    step = 0
    final_answer = ""

    for chunk in agent.stream({"messages": [{"role": "user", "content": query}]}, stream_mode="updates"):
        if "model" in chunk:
            for msg in chunk["model"]["messages"]:
                if getattr(msg, "tool_calls", None):
                    for tc in msg.tool_calls:
                        step += 1
                        cmds = tc["args"].get("commands", [])
                        cmd_str = " && ".join(cmds) if isinstance(cmds, list) else str(cmds)
                        console.print(f"  [step]Step {step}[/] [cmd]{cmd_str}[/]")
                elif msg.content:
                    final_answer = msg.content

        elif "tools" in chunk:
            for msg in chunk["tools"]["messages"]:
                output = msg.content.strip()
                if output:
                    lines = output.splitlines()
                    preview = "\n    ".join(lines[:5])
                    if len(lines) > 5:
                        preview += f"\n    [dim]… {len(lines) - 5} more lines[/dim]"
                    console.print(f"    [output]{preview}[/]")

    return final_answer


def main() -> None:
    registry = SkillRegistry("skills")
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    router = ChatOllama(model="lfm2.5-thinking:1.2b", base_url=ollama_url)
    llm = ChatOllama(model="gemma4:e4b-nvfp4", base_url=ollama_url)

    console.print("\n[bold]Dynamic Skill Agent[/bold] [dim](Ctrl+C to exit)[/dim]\n")

    while True:
        try:
            query = console.input("[bold]Query:[/bold] ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not query:
            continue

        console.print()

        with console.status("[routing]Routing...[/]", spinner="dots"):
            skill_name = route_skill(router, registry.list(), query)
            skill = registry.get(skill_name)

        if skill is None:
            skill = registry.list()[0]
            console.print(f"  [routing]→[/] [dim]unknown '{skill_name}', fallback:[/] [skill]{skill.name}[/]")
        else:
            console.print(f"  [routing]→[/] [skill]{skill.name}[/]")

        console.print()
        answer = run_agent(llm, skill, query)
        console.print()
        console.print(Rule(style="dim"))
        console.print(f"[answer]{answer}[/]")
        console.print()


if __name__ == "__main__":
    main()
