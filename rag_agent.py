import chromadb
import ollama

# Инициализация векторной БД
client = chromadb.Client()
collection = client.get_or_create_collection("codebase")

def add_document(doc_id: str, text: str):
    """Добавляет документ/файл в память"""
    response = ollama.embeddings(model="nomic-embed-text", prompt=text)
    collection.add(
        ids=[doc_id],
        embeddings=[response['embedding']],
        documents=[text]
    )

def search_context(query: str, n_results: int = 3) -> str:
    """Ищет релевантный контекст для запроса"""
    response = ollama.embeddings(model="nomic-embed-text", prompt=query)
    results = collection.query(
        query_embeddings=[response['embedding']],
        n_results=n_results
    )
    return "\n---\n".join(results['documents'][0])

def rag_code_agent(task: str) -> str:
    """Агент с доступом к контексту проекта"""
    context = search_context(task)
    
    response = ollama.chat(
        model="qwen2.5-coder:7b",
        messages=[{
            "role": "system",
            "content": f"Контекст проекта:\n{context}\n\nИспользуй его при ответе."
        }, {
            "role": "user",
            "content": task
        }]
    )
    return response['message']['content']


if __name__ == "__main__":
    # Добавляем документы проекта
    add_document("auth", "def login(user, password): проверяет credentials через JWT")
    add_document("db", "Используем PostgreSQL, модель User: id, email, created_at")
    
    # Запрос с контекстом
    result = rag_code_agent("Напиши функцию регистрации пользователя")
    print(result)