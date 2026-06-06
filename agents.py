import ollama

def code_agent(task: str) -> str:
    """Генерирует код по заданию"""
    response = ollama.chat(
        model="qwen2.5-coder:7b",
        messages=[{
            "role": "system",
            "content": "Ты эксперт-программист. Пиши чистый, рабочий код с комментариями."
        }, {
            "role": "user",
            "content": task
        }]
    )
    return response['message']['content']


def review_agent(task: str) -> str:
    """Ревьюит код, находит баги и улучшения"""
    response = ollama.chat(
        model="qwen2.5:14b",
        messages=[{
            "role": "system",
            "content": """Ты senior code reviewer. Анализируй код и давай структурированный отчёт:
1. БАГИ — критические проблемы
2. УЛУЧШЕНИЯ — что можно сделать лучше  
3. ОЦЕНКА — от 1 до 10"""
        }, {
            "role": "user",
            "content": task
        }]
    )
    return response['message']['content']


def simple_agent(task: str) -> str:
    """Отвечает на простые вопросы"""
    response = ollama.chat(
        model="llama3.2:3b",
        messages=[{
            "role": "user",
            "content": task
        }]
    )
    return response['message']['content']


def complex_agent(task: str) -> str:
    """Решает сложные архитектурные задачи"""
    response = ollama.chat(
        model="qwen2.5:14b",
        messages=[{
            "role": "system",
            "content": "Ты опытный software architect. Давай детальные, структурированные ответы."
        }, {
            "role": "user",
            "content": task
        }]
    )
    return response['message']['content']


# Тест каждого агента
if __name__ == "__main__":
    print("=== CODE AGENT ===")
    print(code_agent("Напиши функцию на Python для поиска простых чисел"))
    
    print("\n=== REVIEW AGENT ===")
    print(review_agent("Проверь этот код:\ndef divide(a, b):\n    return a / b"))