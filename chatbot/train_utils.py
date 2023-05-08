import json
import torch
from torch.utils.data import Dataset
from chatbot.nltk_utils import stem, tokenize, bag_of_words


def save_model(data):
    model_name = "./chatbot/chat_model.pth"
    torch.save(data, model_name)


def load_model(model_name):
    chat_model = torch.load(f"./chatbot/{model_name}")
    return chat_model


def working_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device


def communication_materials():
    materials = {}
    with open("./chatbot/materials.json", 'r') as ptr:
        materials = json.load(ptr)
    return materials


materials = communication_materials()
ignored_characters = ['!', '@', '#', '$', '₹', '%', '^', '&', '*',
                      '(', ')', '-', '+', '=', ',', '.', '|', '`',
                      '~', ';', ':', '?', '<', '>', '{', '}', '[', ']']


def get_tags():
    tags = []
    for i in materials["materials"]:
        tags.append(i["tag"])
    tags = sorted(set(tags))
    return tags


def get_words():
    words = []
    for i in materials["materials"]:
        for pattern in i["patterns"]:
            word_list = tokenize(sentence=pattern)
            words.extend(word_list)
    words = [stem(word) for word in words if word not in ignored_characters]
    words = sorted(set(words))
    return words


def get_xy_dataset():
    xy = []
    for i in materials["materials"]:
        for pattern in i["patterns"]:
            word_list = tokenize(sentence=pattern)
            xy.append((word_list, i["tag"]))
    return xy


def get_xy_train_dataset():
    x_train, y_train = [], []
    tags, words = get_tags(), get_words()
    for (tokenized_sentence, tag) in get_xy_dataset():
        bag = bag_of_words(tokenized_sentence, words)
        x_train.append(bag)
        label = tags.index(tag)
        y_train.append(label)
    return x_train, y_train


class ChatDataset(Dataset):
    def __init__(self, x_train, y_train):
        self.no_of_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.no_of_samples


def train_model(model, criterion, optimizer, train_loader, epochs, device):
    for epoch in range(epochs):
        for (train_words, labels) in train_loader:
            train_words = train_words.to(device)
            labels = labels.to(dtype=torch.long).to(device)

            # forward
            outputs = model(train_words)
            loss = criterion(outputs, labels)

            # backward and optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"epoch: {epoch + 1} / {epochs}, loss: {loss.item()}")

    final_loss = loss.item()
    return final_loss


if __name__ == "__main__":
    # print(get_tags())
    # print(get_words())
    # print(get_xy_dataset())
    # print(get_xy_train_dataset())
    pass
