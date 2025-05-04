import typer
from forge.agents.narrator import run_narrator

app = typer.Typer()


@app.command()
def run(agent: str):
    """Run a specific agent."""
    if agent == "narrator":
        run_narrator()
    else:
        typer.echo(f"Agent '{agent}' not supported yet.")
