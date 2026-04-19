import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Skill:
    name: str
    description: str
    content: str


class SkillRegistry:
    def __init__(self, skills_dir: str = "skills"):
        self.skills: dict[str, Skill] = {}
        self._load(Path(skills_dir))

    def _parse_frontmatter(self, text: str) -> tuple[dict, str]:
        match = re.match(r"^---\n(.*?)\n---\n(.*)", text, re.DOTALL)
        if not match:
            return {}, text
        meta: dict[str, str] = {}
        for line in match.group(1).split("\n"):
            if ":" in line:
                key, _, value = line.partition(":")
                meta[key.strip()] = value.strip()
        return meta, match.group(2).strip()

    def _load(self, path: Path) -> None:
        for file in sorted(path.glob("*.md")):
            meta, content = self._parse_frontmatter(file.read_text())
            skill = Skill(
                name=meta.get("name", file.stem),
                description=meta.get("description", ""),
                content=content,
            )
            self.skills[skill.name] = skill

    def list(self) -> list[Skill]:
        return list(self.skills.values())

    def get(self, name: str) -> Skill | None:
        return self.skills.get(name)
