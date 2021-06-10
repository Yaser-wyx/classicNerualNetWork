from torch.utils.data import DataLoader
from torchvision import transforms

from classify.TinyImagenetLoader import data_process, ImageDataset
from classify.VGG import *
import torch
import copy
import os
from torch.utils.tensorboard import SummaryWriter


def train(model, train_data_loader, valid_data_loader, device, optimizer, loss_fn, epochs, get_best=True):
    writer = SummaryWriter('runs/fashion_mnist_experiment_1')

    best_loss = 99999
    best_model = None
    for epoch in range(epochs):
        model.train()
        for index, (input, labels) in enumerate(train_data_loader):
            input, labels = input.to(device), labels.to(device)
            output = model(input)
            loss = loss_fn(output, labels)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            if index % 10 == 0:
                pred = output.argmax(dim=1)
                acc = torch.sum(pred.flatten() == labels.flatten()).item() * 100
                writer.add_scalar('Loss/train', float(loss), epoch)
                print("Epoch:{} iterations:{} loss:{} acc:{}%".format(epoch, index, loss.item(), acc / len(pred)))
        model.eval()
        total_loss = 0.
        total_acc = 0.
        for index, (input, labels) in enumerate(valid_data_loader):
            input, labels = input.to(device), labels.to(device)
            output = model(input)
            loss = loss_fn(output, labels)
            pred = output.argmax(dim=1)
            pred = pred.view(-1)
            label_flatten = labels.view(-1)
            total_loss += loss.item() * input.shape[0]
            print(label_flatten, pred)
            total_acc += torch.sum(pred == label_flatten).item()
        val_loss = total_loss / len(valid_data_loader.dataset)
        print("Epoch:{} val_loss:{} val_acc:{}%".format(epoch, val_loss,
                                                        (total_acc * 100) / len(valid_data_loader.dataset)))
        if val_loss < best_loss:
            best_model = copy.deepcopy(model)
            best_loss = val_loss
    if get_best:
        return best_model
    else:
        return model


if __name__ == '__main__':
    model = VGG(200)
    root = os.getcwd()
    input_size = 64
    batch_size = 128
    labels_map, idx2label, train_data, val_data = data_process(root)
    train_dataset = ImageDataset(train_data, transforms=transforms.Compose([
        transforms.ToTensor(),

        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ]))
    val_dataset = ImageDataset(val_data, transforms=transforms.Compose([
        transforms.ToTensor()
    ]))

    image_dataloader = {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = 0.0005
    epochs = 50
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.6)
    loss_fn = nn.CrossEntropyLoss()
    model.to(device=device)
    train(model, train_data_loader=image_dataloader["train"], valid_data_loader=image_dataloader["val"], device=device,
          loss_fn=loss_fn,
          optimizer=optimizer, epochs=epochs)
