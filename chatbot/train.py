import torch
from nn_model import nn, NeuralNet
from torch.utils.data import DataLoader
from train_utils import ChatDataset, get_tags, get_words, get_xy_train_dataset, save_model, train_model, working_device, get_xy_dataset


def train():
    print("Training begins...", end="\n\n")
    # required training stuffs
    device = working_device()
    tags, words = get_tags(), get_words()
    x_train, y_train = get_xy_train_dataset()

    # Hyper-Parameters
    batch_size = 8
    hidden_size = 8
    learning_rate = 0.001
    output_size = len(tags)
    input_size = len(x_train[0])  # or len(words)

    # prepare dataset
    dataset = ChatDataset(x_train, y_train)
    train_loader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # making NeuralNet model with device (CPU / GPU)
    model = NeuralNet(input_size, hidden_size, output_size).to(device)

    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # train the NeuralNet model
    final_loss = train_model(model, criterion, optimizer, train_loader,
                             epochs=1000, device=device)

    # save the nn model
    data = {
        "tags": tags,
        "words": words,
        "input_size": input_size,
        "output_size": output_size,
        "hidden_size": hidden_size,
        "model_state": model.state_dict()
    }
    save_model(data)
    print("\nTraining ends...", end="\n\n")


if __name__ == "__main__":
    train()
    pass
