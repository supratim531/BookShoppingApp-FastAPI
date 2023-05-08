import os
import torch
import random
from chatbot.nn_model import NeuralNet
from chatbot.nltk_utils import bag_of_words, tokenize
from chatbot.train_utils import communication_materials, load_model, working_device


def is_training_required():
    return os.path.exists("./chatbot/chat_model.pth") == False


def get_tag_and_probability(query=""):
    # load the trained model
    device = working_device()
    trained_model = load_model(model_name="chat_model.pth")

    tags = trained_model['tags']
    words = trained_model['words']
    input_size = trained_model['input_size']
    output_size = trained_model['output_size']
    hidden_size = trained_model['hidden_size']
    model_state = trained_model['model_state']

    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()

    tokenized_words = tokenize(query)
    X = bag_of_words(tokenized_words, words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
    # print(X)

    output = model(X)
    useless, predicted = torch.max(output, dim=1)
    predicted_tag_index = predicted.item()
    tag = tags[predicted_tag_index]
    # print(tag, output, useless, predicted, predicted_tag_index)

    probabilities = torch.softmax(output, dim=1)
    probability = probabilities[0][predicted_tag_index].item()
    # print(probability, probabilities)

    return tag, probability


def answer_of_chatbot(query=""):
    materials = communication_materials()
    tag, probability = get_tag_and_probability(query)
    # print(materials, tag, probability)

    if probability >= 0.75:
        for i in materials["materials"]:
            if tag == i["tag"]:
                return random.choice(i["responses"])

    return "I don't understand"


if __name__ == "__main__":
    pass
