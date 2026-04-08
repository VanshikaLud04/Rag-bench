#!/bin/bash
echo "Pulling phi3..."
docker exec ragbench-ollama-1 ollama pull phi3

echo "Pulling mistral..."
docker exec ragbench-ollama-1 ollama pull mistral

echo "Done. Models ready."