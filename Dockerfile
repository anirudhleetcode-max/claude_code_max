# AI Engineer in a container. Mount the project to operate on at /workspace.
#   docker build -t ai-engineer .
#   docker run --rm -it -v "$PWD:/workspace" -e ANTHROPIC_API_KEY -e AIE_MODEL ai-engineer run "your task"
# The web dashboard: add -p 8765:8765 and run `ui --host 0.0.0.0` (always set AIE_WEB_TOKEN).
FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ripgrep ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 agent
WORKDIR /opt/ai-engineer
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[anthropic,web]"

USER agent
ENV AIE_HOME=/home/agent/.ai-engineer
RUN git config --global --add safe.directory /workspace
WORKDIR /workspace
ENTRYPOINT ["aie"]
CMD ["--help"]
