from sys import argv
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

def generate_llm_response(input_text: str, temp: float = 1.0) -> None:
    # The client gets the API key from the environment variable `GEMINI_API_KEY`.
    if not input_text:
        raise ValueError("O prompt não pode ser vazio.")
    if temp < 0.0 or temp > 2.01:
        raise ValueError("A temperatura deve estar entre 0.0 e 2.0.")

    client = genai.Client()

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=input_text,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            temperature=temp
        )
    )
    print(response.text)


def main():
    if len(argv) < 2:
        return print("Uso: prompt.py <input_text> [<temperature>]")

    try:
        input_text = argv[1]
        if len(argv) >= 3:
            generate_llm_response(input_text, float(argv[2]))
        generate_llm_response(input_text)

    except ValueError as e:
        return print(f"Error: {e}")



if __name__ == "__main__":
    main()

