from sys import argv
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

def generate_llm_response(input_text: str, temp: float = 2.0) -> str:
    if not input_text:
        raise ValueError("O prompt não pode ser vazio.")

    client = genai.Client()

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=input_text,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            temperature=temp,
        )
    )

    return response.text

def main():
    try:
        if len(argv) < 2:
            raise ValueError("Uso: chaining.py <texto>")
        input_text = argv[1]
        prompt1 = f"Gere uma descrição mais detalhada e persuasiva do produto:\n{input_text}\n com cerca de 50 palavras."
        response1 = generate_llm_response(prompt1)
        print(f"Descrição detalhada do produto:\n{response1}\n")

        prompt2 = f"Converta a descrição a seguir em um único anúncio publicitário curto e criativo:\n{response1}\n use emojis"
        response2 = generate_llm_response(prompt2)
        print(f"Anúncio publicitário:\n{response2}\n")

        prompt3 = f"Traduza o seguinte texto para o inglês:\n{response2}\n e responda apenas com a tradução."
        response3 = generate_llm_response(prompt3)
        print(f"Tradução para o inglês:\n{response3}\n")
    except ValueError as e:
        return print(f"Error: {e}")

if __name__ == "__main__":
    main()