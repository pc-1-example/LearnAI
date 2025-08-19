import os
from dotenv import load_dotenv
known_working_key = ""
print("--- ГЛУБОКАЯ ДИАГНОСТИКА КЛЮЧЕЙ ---")
load_dotenv()
key_from_env = os.getenv("GEMINI_API_KEY")
if not key_from_env:
    print("\n❌ ОШИБКА: Не удалось найти ключ GEMINI_API_KEY в файле .env.")
else:
    print("\n[1] Прямое сравнение строк:")
    if key_from_env == known_working_key:
        print("✅ Строки ПОЛНОСТЬЮ идентичны. Это очень странно.")
    else:
        print("❌ ВНИМАНИЕ: Строки РАЗЛИЧАЮТСЯ! Это и есть причина проблемы.")
    print("\n[2] Сравнение по длине:")
    print(f"   Длина ключа из .env:      {len(key_from_env)}")
    print(f"   Длина рабочего ключа:     {len(known_working_key)}")
    print("\n[3] Сравнение байтового представления (покажет скрытые символы):")
    print(f"   Байты из .env:      {key_from_env.encode('utf-8')}")
    print(f"   Байты рабочего ключа: {known_working_key.encode('utf-8')}")
    print("\nСравните выводы выше. Если байты отличаются (например, наличием `\\r`), вы нашли проблему.")
print("\n--- КОНЕЦ ДИАГНОСТИКИ ---")