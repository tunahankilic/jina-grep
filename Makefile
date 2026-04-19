.PHONY: build run

build:
	docker build -t jina-grep .

run:
	docker run -it \
		--read-only \
		--tmpfs /tmp \
		--tmpfs /home/agent/.cache \
		--cap-drop ALL \
		--security-opt no-new-privileges \
		-e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
		-v $(PWD)/data:/app/data:ro \
		jina-grep
