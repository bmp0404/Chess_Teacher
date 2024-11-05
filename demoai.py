from Agents import *


def getMove(fen):
    agent = SortedTranspositionAgent()

    return str(agent.get_move(fen))


# def main():
#     getMove("8/3p1k2/2p2p2/2P2P2/3P1P2/1P6/4K3/8 w - - 0 1")

# main()