from train import train
from chat import answer_of_chatbot as answer, is_training_required


def chat_is_over(query=""):
    return True if query.lower() in ['e', 'q', "exit", "quit"]else False


def chatbot():
    # checking chat_model.pth existance
    if is_training_required():
        train()

    print("Let's chat! Type e, q, exit or quit to exit\n")
    while True:
        query = input(f"You: ")

        if chat_is_over(query):
            print("Bot: Thanks For Using! Have A Good Day Dear...")
            break

        print(f"Bot: {answer(query)}", end="\n\n")


if __name__ == "__main__":
    chatbot()
