import os
from dotenv import load_dotenv
load_success = load_dotenv()

print("--- НАЧАЛО ДИАГНОСТИКИ ---")
if load_success:
    print("✅ Файл .env найден и успешно загружен.")
else:
    print("❌ ВНИМАНИЕ: Файл .env не найден в этой директории!")
api_key = os.getenv("GEMINI_API_KEY")

print(f"Имя переменной для поиска: 'GEMINI_API_KEY'")
if api_key:
    print("✅ Переменная GEMINI_API_KEY найдена!")
    print(f"   Длина ключа: {len(api_key)} символов.")
    print(f"   Ключ, который видит программа: >>>{api_key}<<<")
    if " " in api_key:
        print("   ❌ ВНИМАНИЕ: В ключе обнаружены пробелы!")
else:
    print("❌ ВНИМАНИЕ: Переменная GEMINI_API_KEY не найдена в окружении!")
print("--- КОНЕЦ ДИАГНОСТИКИ ---")