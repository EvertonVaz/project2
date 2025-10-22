from sys import argv
from google import genai
from google.genai import types
from dotenv import load_dotenv
from pydantic import BaseModel

class Pessoa(BaseModel):
    nome: str
    idade: int
    profissão: str
    cidade: str

load_dotenv()

def generate_xml(content: str) -> str:
    xml_template = """
    <instruções>
    1. extraia as informações da pessoa, nome, idade, profissão e cidade.
    2. responda em formato JSON.
    </instruções>

    Siga esta estrutura e responda sem as tags XML:
    <exemplos>
        <exemplo>
            <texto>"João Silva, 35 anos, é engenheiro e mora em SP"</texto>
            <resposta>
            {"nome": "João Silva", "idade": "35", "profissão": "engenheiro", "cidade": "São Paulo"}
            </resposta>
        </exemplo>
        <exemplo>
            <texto>"Maria Oliveira, 28 anos, trabalha como médica em RJ"</texto>
            <resposta>
            {"nome": "Maria Oliveira", "idade": "28", "profissão": "médica", "cidade": "Rio de Janeiro"}
            </resposta>
        </exemplo>
    </exemplos>
    """
    xml_template += f"<texto> {content} </texto>"
    return xml_template


def generate_llm_response(input_text: str, temp: float = 0.0) -> str:
    if not input_text:
        raise ValueError("O prompt não pode ser vazio.")

    client = genai.Client()
    prompt = generate_xml(input_text)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            temperature=temp,
            response_mime_type="application/json",
            response_schema=Pessoa,
        )
    )

    response:Pessoa = response.parsed
    return response

def main():
    try:
        if len(argv) < 2:
            raise ValueError("Uso: structured.py <texto>")
        input_text = argv[1]

        response = generate_llm_response(input_text)
        print(f"{response}")
    except ValueError as e:
        return print(f"Error: {e}")

if __name__ == "__main__":
    main()