import ollama

ROUTER_PROMPT = """
Ты классификатор задач. Получаешь задачу и отвечаешь ТОЛЬКО одним словом:

- SIMPLE   → простой вопрос, объяснение, документация
- CODE     → написать новый код, функцию, класс
- REVIEW   → проверить код, найти баги, улучшить код
- COMPLEX  → архитектура, дизайн системы, сложный анализ

Задача: {task}
Ответ (одно слово):
"""

def route_task(task: str) -> str:
    response = ollama.chat(
        model="llama3.2:3b",
        messages=[{
            "role": "user",
            "content": ROUTER_PROMPT.format(task=task)
        }]
    )
    result = response['message']['content'].strip().upper()
    
    # Нормализация ответа
    for category in ["SIMPLE", "CODE", "REVIEW", "COMPLEX"]:
        if category in result:
            return category
    return "SIMPLE"  # fallback


# Тест
if __name__ == "__main__":
    test_tasks = [
        "Напиши функцию сортировки пузырьком",
        "Найди баги в этом коде: def foo(x): return x/0",
        "Как работает Git rebase?",
        "Спроектируй микросервисную архитектуру для интернет-магазина"
    ]
    
    for task in test_tasks:
        category = route_task(task)
        print(f"[{category}] {task[:50]}")