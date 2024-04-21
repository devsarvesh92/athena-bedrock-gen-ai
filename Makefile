generate-vector-embedings:
	@pdm run python src/embed.py
	$(call teardown)

local-setup:
	@cp -f .env.template .env
	@pdm config python.use_venv true
	@pdm install --no-self

start-query-genration:
	@pdm run python src/app.py
