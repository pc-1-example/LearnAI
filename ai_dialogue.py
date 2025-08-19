import google.generativeai as genai
import os
import time
from dotenv import load_dotenv
from colorama import init, Fore, Style

init(autoreset=True)

def main():
    try:
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print(Fore.RED + "Ошибка: API ключ не найден. Убедитесь, что он есть в файле .env")
            return
        genai.configure(api_key=api_key)
    except Exception as e:
        print(Fore.RED + f"Произошла ошибка при настройке API: {e}")
        return

    model = genai.GenerativeModel('gemma-3-27b-it')
    chat1 = model.start_chat(history=[])
    chat2 = model.start_chat(history=[])

    print(Style.BRIGHT + Fore.YELLOW + "--- Начало диалога двух ИИ ---")
    print(Style.DIM + "Нажмите Ctrl+C, чтобы остановить.\n")
    time.sleep(2)

    next_message = "Привет"
    print(f"{Fore.WHITE}Стартовое сообщение: {next_message}\n" + "-"*20)
    time.sleep(1)

    try:
        while True:
            response1 = chat1.send_message(next_message)
            next_message = response1.text
            print(f"{Fore.CYAN}{Style.BRIGHT}Gemini 1:{Style.RESET_ALL} {next_message}")
            time.sleep(2) 

            response2 = chat2.send_message(next_message)
            next_message = response2.text
            print(f"{Fore.MAGENTA}{Style.BRIGHT}Gemini 2:{Style.RESET_ALL} {next_message}")
            print(Style.DIM + "...")
            time.sleep(3) 

    except KeyboardInterrupt:
        print(Style.BRIGHT + Fore.YELLOW + "\n\n--- Диалог прерван пользователем ---")
    except Exception as e:
        print(Fore.RED + f"\nПроизошла критическая ошибка: {e}")


if __name__ == "__main__":
    main()