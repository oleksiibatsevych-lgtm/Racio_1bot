import google.generativeai as genai

def generate_ai_audit(prompt_text: str) -> str:
    """
    Виконує запит до Gemini з підтримкою кількох запасних моделей (fallback).
    """
    # Список моделей у порядку пріоритету: від нової до резервних
    models_to_try = [
        "gemini-3.6-flash",
        "gemini-2.5-flash",
        "gemini-1.5-flash"
    ]
    
    for model_name in models_to_try:
        try:
            # Ініціалізація моделі
            model = genai.GenerativeModel(model_name)
            
            # Генерація відповіді
            response = model.generate_content(prompt_text)
            
            if response and response.text:
                return response.text
                
        except Exception as e:
            # Логуємо помилку для конкретної моделі і переходимо до наступної
            print(f"⚠️ Модель {model_name} недоступна: {e}")
            continue
            
    # Якщо жодна модель не спрацювала
    print("❌ Помилка AI Audit: Усі моделі Gemini наразі недоступні.")
    return None
