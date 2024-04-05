from langchain_community.llms import Ollama
def main():
    ollama = Ollama(base_url="http://localhost:11434", model="llama3")

    TEXT_PROMPT = "You are a chess coach. I am going to give you a PGN with each move having a comment with the evaluation. I am white, and my opponent is black. How can I improve?"

    with open('game.pgn', 'r') as file:
        file_contents = file.read()

    TEXT_PROMPT = TEXT_PROMPT + file_contents + "Provide general notes for White with examples and an overview summary"


    print("thinking")
    print(ollama(TEXT_PROMPT))