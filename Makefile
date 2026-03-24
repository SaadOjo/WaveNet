COMPOSE := docker compose
SERVICE := app

.PHONY: build up down restart shell logs ps run xhost

build:
	$(COMPOSE) build

xhost:
	xhost +local:docker

up: xhost
	$(COMPOSE) up -d

down:
	$(COMPOSE) down

restart: down up

shell:
	$(COMPOSE) exec $(SERVICE) bash

logs:
	$(COMPOSE) logs -f $(SERVICE)

ps:
	$(COMPOSE) ps

run:
	@if [ -z "$(CMD)" ]; then \
		echo "Usage: make run CMD='python path/to/script.py --arg value'"; \
		exit 1; \
	fi
	$(COMPOSE) exec $(SERVICE) bash -lc "$(CMD)"
