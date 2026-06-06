import ollama

# Проверяем связь с каждой моделью
models = ["llama3.2:3b", "qwen2.5-coder:7b", "qwen2.5:14b"]

for model in models:
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": "Ответь одним словом: работаешь?"}]
    )
    print(f"{model}: {response['message']['content']}")