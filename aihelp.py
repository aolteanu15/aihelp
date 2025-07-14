import typer
from rich import print
import os
from openai import OpenAI
from typing import List

app = typer.Typer()

@app.command()
def ask(question: List[str] = typer.Argument(..., help="Your question")):
    """Ask a question using Groq's LLaMA3 model."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("[red]Error:[/red] GROQ_API_KEY not set.")
        return

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1"
    )

    full_question = " ".join(question)

    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": full_question}]
        )
        print("[bold green]Answer:[/bold green]", response.choices[0].message.content.strip())
    except Exception as e:
        print(f"[red]Request failed:[/red] {e}")

if __name__ == "__main__":
    app()
