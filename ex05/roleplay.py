from sys import argv
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

def generate_xml(content: str) -> str:
    xml_template = f"""
    <persona>
    Você é um consultor de viagens especializado no destino {content}, e conhece bem os melhores pontos turísticos da região.
    </persona>

    <instruções>
    1. Liste os 5 melhores destinos turísticos da região.
    2. Numero os destinos de 1 a 5.
    </instruções>

    Siga esta estrutura e responda sem as tags XML:
    <exemplos>
        <exemplo>
            <destino>Paris</destino>
            <resposta>
            1. Torre Eiffel
            2. Museu do Louvre
            3. Catedral de Notre-Dame
            4. Arco do Triunfo
            5. Basílica de Sacré-Cœur
            </resumo>
        </exemplo>
        <exemplo>
            <destino>Floresta Amazônica</destino>
            <resposta>
            1. Encontro das Águas
            2. Parque Nacional de Anavilhanas
            3. Reserva Adolpho Ducke
            4. Lago Janauari
            5. Comunidade de Mamirauá
            </resposta>
        </exemplo>
    </exemplos>

    <destino> {content} </destino>
    """
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
            temperature=temp
        )
    )

    return response.text

def main():
    try:
        if len(argv) < 2:
            raise ValueError("Uso: roleplay.py <destino>")
        input_text = argv[1]

        response = generate_llm_response(input_text)
        print(f"{response}")
    except ValueError as e:
        return print(f"Error: {e}")

if __name__ == "__main__":
    main()