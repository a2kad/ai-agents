from router_agent import route_task
from agents import code_agent, review_agent, simple_agent, complex_agent
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()

AGENT_MAP = {
    "SIMPLE":  (simple_agent,  "llama3.2:3b",    "cyan"),
    "CODE":    (code_agent,    "qwen2.5-coder:7b","green"),
    "REVIEW":  (review_agent,  "qwen2.5:14b",     "yellow"),
    "COMPLEX": (complex_agent, "qwen2.5:14b",     "red"),
}

def run_pipeline(task: str) -> str:
    # Шаг 1: Роутинг
    console.print(f"\n[bold]📥 Задача:[/bold] {task}")
    category = route_task(task)
    
    agent_func, model_name, color = AGENT_MAP[category]
    console.print(f"[{color}]🔀 Роутер → {category} [{model_name}][/{color}]")
    
    # Шаг 2: Выполнение
    console.print(f"[{color}]⚙️  Обработка...[/{color}]")
    result = agent_func(task)
    
    # Шаг 3: Если это код — автоматически отправить на ревью
    if category == "CODE":
        console.print("[yellow]🔍 Автоматическое ревью сгенерированного кода...[/yellow]")
        review = review_agent(f"Проверь этот код:\n{result}")
        
        console.print(Panel(Markdown(result), title="✅ Сгенерированный код", border_style="green"))
        console.print(Panel(Markdown(review), title="🔍 Ревью", border_style="yellow"))
        return result
    
    console.print(Panel(Markdown(result), title=f"✅ Ответ [{model_name}]", border_style=color))
    return result


# Интерактивный режим
if __name__ == "__main__":
    console.print(Panel("[bold green]🤖 AI Code Pipeline[/bold green]\nВведите 'exit' для выхода", 
                        border_style="green"))
    
    while True:
        task = console.input("\n[bold cyan]Ваша задача:[/bold cyan] ")
        if task.lower() == "exit":
            break
        run_pipeline(task)