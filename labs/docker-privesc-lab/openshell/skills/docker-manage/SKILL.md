# docker-manage — Agent Skill

Routes Docker container management tasks from inside an OpenShell sandbox to
the Docker Manager MCP server running outside it.

## Tools

| Tool | Description |
|---|---|
| `list_containers` | List running containers (benign) |
| `run_named_compose` | Run a pre-approved compose file by name (validated) |
| `run_container` | Run a container with caller-supplied config |
| `apply_compose` | Apply a compose YAML supplied as a string |

## Security note

`run_container` and `apply_compose` are deliberately vulnerable (CWE-269).
They demonstrate that an agent with access to a Docker management MCP server
can reach root on the host without `sudo`, without writing a privileged compose
file to disk, and without heredoc patterns visible to the host.

The `run_named_compose` tool is a negative control: it validates for
`privileged: true` and host root mounts before running — showing that keyword
detection catches the naive file-write attack (Path B) but does not protect
against the inline-param (Path C) or stdin-pipe (Path D) variants.
