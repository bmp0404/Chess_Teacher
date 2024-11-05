from langchain_community.llms import Ollama
# side by default will be white and name will 
def main(side = 0):
    ollama = Ollama(base_url="http://localhost:11434", model="llama3")

    temp = "black" if side == 1 else "white"
    opponet = "black" if side != 1 else "white"

    TEXT_PROMPT = "You are a chess coach. I am going to give you a PGN with each move having a comment with the evaluation. I am " + temp + ", and my opponent is "+ opponet +". How can I improve?"

    with open('game.pgn', 'r') as file:
        file_contents = file.read()

    TEXT_PROMPT = TEXT_PROMPT + file_contents + "Provide general notes for " + temp + " with examples and an overview summary"


    print("Thinking:")
    print(ollama(TEXT_PROMPT))