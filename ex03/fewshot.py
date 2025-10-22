from sys import argv
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

def build_few_shot_prompt(input_text: str) -> str:
    examples = [
        "SQL\nlinguagem de consulta estruturada.",
        "Python\numa linguagem de programação de alto nível.",
        "API\nconjunto de definições e protocolos para construir e integrar software.",
        "HTTP\nprotocolo de transferência de hipertexto, usado para comunicação na web."
    ]
    prompt = "Aqui estão alguns exemplos de perguntas e respostas, responda sem as perguntas:\n\n"
    prompt += "\n\n".join(examples)
    prompt += f"\n\n{input_text}\n"
    return prompt

def generate_llm_response(input_text: str, temp: float = 1.0) -> None:
    if not input_text:
        raise ValueError("O prompt não pode ser vazio.")

    client = genai.Client()
    few_shot_prompt = build_few_shot_prompt(input_text)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=few_shot_prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            temperature=temp
        )
    )
    print(response.text)


def main():
    try:
        if len(argv) < 2:
            raise ValueError("Uso: fewshot.py <input_text>")
        input_text = argv[1]

        generate_llm_response(input_text)
    except ValueError as e:
        return print(f"Error: {e}")



if __name__ == "__main__":
    main()

