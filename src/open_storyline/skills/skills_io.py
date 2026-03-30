

import aiofiles
from pathlib import Path
from skillkit import SkillManager
from skillkit.integrations.langchain import create_langchain_tools
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage

async def load_skills(
    skill_dir: str | Path =".storyline/skills"
):
    # Discover skills
    manager = SkillManager(skill_dir=skill_dir)
    await manager.adiscover()

    # Convert to LangChain tools
    tools = create_langchain_tools(manager)
    return tools

async def dump_skills(
    skill_name: str = '',
    skill_dir: str = '',
    skill_content: str = '',
    **kwargs,
):
    
    clean_name = skill_name.strip()
    if not clean_name:
        return {
            "status": "error",
            "message": "skill_name cannot be empty"
        }

    base_path = Path.cwd()

    # Project_Root + skill_dir + skill_name/
    target_path = base_path / skill_dir / f"cutskill_{clean_name}"

    # Fix name: SKILL.md
    target_file_path = target_path / "SKILL.md"

    # Path Traversal Protection 
    try:
        final_path = target_file_path.resolve()
        if base_path not in final_path.parents:
            return {
                "status": "error",
                "message": f"Security Alert: Writing to paths outside the project directory is forbidden: {final_path}"
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Path resolution error: {str(e)}"
        }

    # Start write
    try:
        if not target_path.exists():
            target_path.mkdir(parents=True, exist_ok=True)

        async with aiofiles.open(final_path, mode='w', encoding='utf-8') as f:
            await f.write(skill_content)

        return {
            "status": "success",
            "message": f"Skill '{clean_name}' successfully created.",
            "dir_path": str(target_path),
            "file_path": str(final_path),
            "size_bytes": len(skill_content.encode('utf-8'))
        }

    except PermissionError:
        return {
            "status": "error",
            "message": f"Permission denied: Cannot write to directory {target_path}"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Write operation failed: {str(e)}"
        }
