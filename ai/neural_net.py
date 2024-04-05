import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os

# Create a custom neural network class
import torch.nn as nn
import torch.nn.functional as F

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(189, 256)  # Input size 189, output size 256
        self.fc2 = nn.Linear(256, 128)     # Input size 255, output size 160
        self.fc3 = nn.Linear(128, 64)     # Input size 160, output size 80
        self.fc4 = nn.Linear(64, 32)     # Input size 80, output size 32
        self.fc5 = nn.Linear(32, 1)     # Input size 32, output size 1
        
    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        x = nn.functional.relu(self.fc4(x))
        x = self.fc5(x)
        return x

# Function to train the neural network
def train_net(input_arr, eval_arr):
    if torch.cuda.is_available(): 
        dev = torch.device("cuda:0")
        print("Using GPU") 
    else: 
        dev = torch.device("cpu") 
        print("Using CPU") 

    input_tensor = torch.tensor(input_arr, dtype=torch.float).to(dev)
    eval_tensor = torch.tensor(eval_arr, dtype=torch.float).view(-1, 1).to(dev)  # Move to GPU
    criterion = nn.MSELoss()
    dataset = TensorDataset(input_tensor, eval_tensor)
    dataloader = DataLoader(dataset, batch_size=16384*2*2, shuffle=True)  # Use DataLoader for batch loading

    num_epochs = 50  # Increase the number of epochs
    model = SimpleNN().to(dev)  # Move model to GPU
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        print("epoch num " + str(epoch))
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item()
    }, 'model_checkpoint.pth')

# Main function
def main():
    print("test")
    pos_list = []
    eval_list = []
    evaluations_file = "pos.txt"
    eval_file = "ai/evaluations.txt"
    
    # Check if both files exist
    if not (os.path.exists(evaluations_file) and os.path.exists(eval_file)):
        print("Files not found.")
        return

    counter = 0  # Initialize counter
    with open(evaluations_file, "r") as f, open(eval_file, "r") as g:
        for line_pos, line_eval in zip(f, g):
            if counter >= 100:
                break  # Break out of the loop if the counter exceeds 1000
            temp = line_pos.strip().split(",")  # Assuming pos.txt contains comma-separated values
            net_input = [float(x.strip().strip('[]')) for x in temp]  # Clean up extra characters and convert to float
            if line_eval[0] == "m":
                continue
            pos_list.append(net_input)
            eval_list.append(float(line_eval))  # Assuming evaluations.txt contains float values
            counter += 1  # Increment counter
            if counter % 100000 == 0:
                print(counter) 
        torch.save(pos_list,"postensor")
        torch.save(eval_list,"evaltensor")

    # pos_list = torch.load("postensor")
    # eval_list = torch.load("evaltensor")
    train_net(pos_list, eval_list)
    
if __name__ == "__main__":
    main()
    # load_checkpoint()
