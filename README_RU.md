# 🤖 AI Code Pipeline — Apple M3 Pro

Локальный многоагентный конвейер разработки кода на базе Ollama. Все модели работают на вашем MacBook — данные не покидают машину.

---

## Архитектура

```
Задача пользователя
        │
        ▼
┌───────────────────┐
│  Роутер-агент     │  llama3.2:3b  — классифицирует задачу
└────────┬──────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
SIMPLE      CODE ──────────────────────┐
  │           │                        │
  ▼           ▼                        ▼
llama3.2   qwen2.5-coder:7b      REVIEW / COMPLEX
  :3b      генерация кода         qwen2.5:14b
                │                 анализ и ревью
                ▼
         Авто-ревью
         qwen2.5:14b
```

---

## Модели

| Модель | Роль | VRAM / RAM |
|---|---|---|
| `llama3.2:3b` | Роутер, классификатор, простые вопросы | ~2 GB |
| `qwen2.5-coder:7b` | Генерация кода | ~4.7 GB |
| `qwen2.5:14b` | Ревью, архитектура, сложный анализ | ~8.9 GB |
| `nomic-embed-text` | Векторизация для RAG | ~0.3 GB |

---

## Требования

- Apple M3 Pro, 18 GB RAM
- macOS 13+
- Python 3.11+
- [Ollama](https://ollama.com) установлен и запущен

---

## Установка

### 1. Ollama и модели

```bash
# Установить Ollama с ollama.com, затем:
ollama pull llama3.2:3b
ollama pull qwen2.5-coder:7b
ollama pull qwen2.5:14b
ollama pull nomic-embed-text
```

### 2. Python-окружение

```bash
git clone <your-repo>
cd ai-pipeline

python3 -m venv venv
source venv/bin/activate

pip install ollama langgraph langchain-ollama chromadb rich
```

---

## Структура проекта

```
ai-pipeline/
├── venv/
├── test_connection.py   # Проверка связи со всеми моделями
├── router_agent.py      # Классификатор задач (llama3.2:3b)
├── agents.py            # Специализированные агенты
├── pipeline.py          # Основной конвейер — точка входа
└── rag_agent.py         # RAG-агент с памятью на проект
```

---

## Запуск

### Интерактивный режим

```bash
source venv/bin/activate
python pipeline.py
```

Введите задачу на русском или английском — конвейер сам выберет нужного агента.

### Примеры запросов

```
Ваша задача: Напиши класс для работы с очередью на Python
→ Роутер: CODE → qwen2.5-coder:7b → авто-ревью qwen2.5:14b

Ваша задача: Найди баги: def divide(a, b): return a / b
→ Роутер: REVIEW → qwen2.5:14b

Ваша задача: Что такое декоратор в Python?
→ Роутер: SIMPLE → llama3.2:3b

Ваша задача: Спроектируй архитектуру микросервисов для интернет-магазина
→ Роутер: COMPLEX → qwen2.5:14b
```

### Проверка работоспособности

```bash
python test_connection.py
# Ожидаемый вывод:
# llama3.2:3b: работаю
# qwen2.5-coder:7b: работаю
# qwen2.5:14b: работаю
```

---

## RAG — память на ваш проект

Индексирует кодовую базу, чтобы агенты знали ваш проект и генерировали код в вашем стиле.

```bash
python rag_agent.py
```

Добавить документы вручную:

```python
from rag_agent import add_document

add_document("auth",   "JWT аутентификация, модель User: id, email, role")
add_document("db",     "PostgreSQL, ORM SQLAlchemy, миграции Alembic")
add_document("api",    "FastAPI, все эндпоинты возвращают {data, error, meta}")
```

После индексации агент учитывает ваши соглашения при генерации кода.

---

## Git Hook — авто-ревью перед коммитом

Устанавливается один раз, запускается автоматически на каждый `git commit`.

```bash
# В корне вашего проекта:
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
cd /path/to/ai-pipeline && source venv/bin/activate
git diff --cached --name-only --diff-filter=M | grep "\.py$" | while read file; do
    python -c "
from agents import review_agent
import sys
code = open('$file').read()
result = review_agent(f'Проверь код:\n{code}')
print(f'\n[REVIEW] $file\n{result}')
if 'оценка: [1-4]' in result.lower():
    sys.exit(1)
"
done
EOF
chmod +x .git/hooks/pre-commit
```

---

## Производительность на M3 Pro

| Модель | Скорость | Время на типичный запрос |
|---|---|---|
| `llama3.2:3b` (роутер) | ~70 tok/s | < 1 сек |
| `qwen2.5-coder:7b` | ~35 tok/s | 5–15 сек |
| `qwen2.5:14b` (ревью) | ~18 tok/s | 15–40 сек |

Unified memory M3 Pro позволяет держать все модели загруженными одновременно без перезагрузки между запросами.

---

## Расширение конвейера

### Добавить нового агента

```python
# в agents.py
def test_agent(task: str) -> str:
    response = ollama.chat(
        model="qwen2.5-coder:7b",
        messages=[{
            "role": "system",
            "content": "Ты эксперт по тестированию. Пиши pytest тесты с edge cases."
        }, {
            "role": "user", "content": task
        }]
    )
    return response['message']['content']
```

```python
# в pipeline.py — добавить в AGENT_MAP:
"TEST": (test_agent, "qwen2.5-coder:7b", "blue"),
```

### Подключить другую модель

```bash
ollama pull deepseek-r1:14b   # сильный в рассуждениях
ollama pull phi4:14b          # компактный и быстрый
```

Заменить `model=` в нужном агенте в `agents.py`.

---

## Интеграция с VS Code

Установить расширение [Continue](https://continue.dev), в настройках указать локальный Ollama:

```json
{
  "models": [
    {
      "title": "Qwen Coder",
      "provider": "ollama",
      "model": "qwen2.5-coder:7b"
    }
  ]
}
```

После этого агент доступен прямо в редакторе через `Cmd+I`.

---

## Лицензия

MIT — используйте свободно.

---

*Построено на: Ollama · Qwen2.5 · Llama 3.2 · LangGraph · ChromaDB*