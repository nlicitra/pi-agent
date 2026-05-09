# Pi Agent Config

Personal Pi agent configuration for skills, extensions, memories, prompts, sessions, and settings.

## Extensions

- `extensions/memory.ts` — Markdown memory system. Loads global/project memory files, injects active memories into agent context, adds `/memory` commands, and blocks suspected secrets in memory writes.

## Memories

- `memories/` contains global Markdown memories loaded into every Pi session.
- Project memories live at `<project-root>/.pi/memories/` and load only inside that project.
- Memory files may use frontmatter like `title`, `autoload`, and `tags`.
- Do not store secrets here; reference environment variable names instead.
- Current memory: `memories/role-names.md` — call the user “human”; “computer” means the Pi coding agent.

Useful commands:

```txt
/memory list
/memory list --all
/memory tags
/memory load <tag>
/memory unload <tag>
/memory reload
/memory add global <name>
/memory add project <name>
/memory edit global <name>
/memory edit project <name>
```
