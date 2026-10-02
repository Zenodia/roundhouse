---
name: memory-recall
description: >-
  Store and recall long-term user memories via the team's Memory MCP server.
  Use this skill whenever the user asks you to remember something, recall a past
  fact, or personalize a response based on prior conversations. Works by calling
  the remote Memory MCP server exposed to this sandbox.
---

# memory-recall

This skill wraps an **MCP client** that talks to the **Memory MCP server**
running *outside* this sandbox. It is the in-sandbox surface that the OpenClaw
agent uses to persist and recall user memories.

## When to use

- The user says "remember that ...", "what do you know about me", "recall ...".
- You need personalization context from earlier sessions.

## How to use

Run the bundled client with a natural-language query and a user id:

```bash
python recall.py --tool memory_agent --query "remember my favorite food is pizza" --user-id ruth
python recall.py --tool fetch_memory_items --query "what food do I like" --user-id ruth
```

Available tools (forwarded to the MCP server):

| Tool                 | Purpose                                             |
| -------------------- | --------------------------------------------------- |
| `memory_agent`       | Chat with memory (stores + answers).                |
| `restart_memory_agent` | Clear memory then chat.                           |
| `fetch_memory_items` | Return raw recalled memory items for a query.       |

## Configuration

The server URL is read from the `MEMORY_MCP_URL` environment variable and
defaults to `http://host.openshell.internal:8999/mcp` (the host that runs the
Memory MCP server, outside this sandbox).

## Security note (this is a deliberately vulnerable lab)

This skill is the entry point used to demonstrate a **sandbox breakout**. The
`user_id` and `query` arguments are forwarded verbatim to the server, which
contains injected vulnerabilities (CWE-78 and CWE-94). See
`exploits/` in the repo for the exploit walk-throughs.
