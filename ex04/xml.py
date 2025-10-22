from sys import argv
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

def generate_xml(content: str) -> str:
    xml_template = f"""
    Você é um resumidor de texto que resume qualquer texto em no maximo 20 palavras.

    <instructions>
    1. Resuma o texto fornecido em no máximo 20 palavras.
    2. Forneça um resumo claro e conciso.
    3. Evite incluir informações irrelevantes ou detalhes excessivos.
    </instructions>

    Siga esta estrutura e responda sem as tags XML:
    <exemplos>
        <exemplo>
            <texto> A inteligência artificial (IA) refere-se à simulação de processos de inteligência humana por máquinas, especialmente sistemas computacionais. </texto>
            <resumo> IA simula processos humanos como aprendizado e tomada de decisão por meio de máquinas. </resumo>
        </exemplo>
        <exemplo>
            <texto> A fotossíntese é o processo pelo qual as plantas, algas e algumas bactérias convertem a luz solar em energia química, produzindo oxigênio como subproduto. </texto>
            <resumo> Fotossíntese converte luz solar em energia química, liberando oxigênio. </resumo>
        </exemplo>
    </exemplos>

    <texto> {content} </texto>
    """
    return xml_template


def generate_llm_response(input_text: str, temp: float = 0.0) -> None:
    if not input_text:
        raise ValueError("O prompt não pode ser vazio.")

    client = genai.Client()
    prompt = generate_xml(input_text)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            temperature=temp
        )
    )
    print(f"\n{response.text}")

def main():
    try:
        if len(argv) < 2:
            raise ValueError("Uso: xml.py <input_text>")
        input_text = argv[1]

        generate_llm_response(input_text)
    except ValueError as e:
        return print(f"Error: {e}")

if __name__ == "__main__":
    main()