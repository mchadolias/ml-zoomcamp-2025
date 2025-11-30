import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot loss
    ax1.plot(history["train_loss"], label="Training")
    ax1.plot(history["val_loss"], label="Validation")
    ax1.set_title("Model Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(frameon=False)

    # Plot accuracy
    ax2.plot(history["train_acc"], label="Training")
    ax2.plot(history["val_acc"], label="Validation")
    ax2.set_title("Model Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend(frameon=False)

    plt.tight_layout()
    plt.show()


from tqdm.auto import tqdm


def train_model(
    train_loader,
    val_loader,
    device,
    model,
    optimizer,
    criterion,
    num_train,
    num_val,
    num_epochs=10,
    scheduler=None,
    early_stopping_patience=5,
):
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "learning_rates": [],
    }

    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0

    # Print table header
    print("\n" + "=" * 75)
    print(
        f"{'Epoch':^6} | {'Train Loss':^12} | {'Train Acc':^10} | {'Val Loss':^12} | {'Val Acc':^10} | {'LR':^10}"
    )
    print("=" * 75)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_idx, (images, labels) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
        ):
            images, labels = images.to(device), labels.to(device)
            labels = labels.float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        # Calculate training metrics
        epoch_train_loss = running_loss / num_train
        epoch_train_acc = correct_train / total_train

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                labels = labels.float().unsqueeze(1)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        # Calculate validation metrics
        epoch_val_loss = val_running_loss / num_val
        epoch_val_acc = correct_val / total_val

        # Update learning rate scheduler if provided
        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler:
            scheduler.step()

        # Store history
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)
        history["learning_rates"].append(current_lr)

        # Print epoch results in table format
        print(
            f"{epoch+1:^6} | {epoch_train_loss:^12.4f} | {epoch_train_acc:^10.4f} | "
            f"{epoch_val_loss:^12.4f} | {epoch_val_acc:^10.4f} | {current_lr:^10.6f}"
        )

        # Early stopping and model checkpointing
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("-" * 75)
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    print("=" * 75)

    # Load best model weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return history, best_val_acc


class HairTypeClassifier(nn.Module):
    def __init__(self, num_classes=1):  # 1 neuron for output as specified
        super(HairTypeClassifier, self).__init__()

        # Convolutional layer: 32 filters, (3,3) kernel
        self.conv_layer = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.relu = nn.ReLU()

        # Max pooling: (2,2) pooling size
        self.max_pool = nn.MaxPool2d(2)

        # Linear layer with 64 neurons
        self.inner = nn.Linear(32 * 99 * 99, 64)  # 313632 neurons after flattening

        # Output layer with 1 neuron and sigmoid activation for binary classification
        self.output_layer = nn.Linear(64, num_classes)

    def forward(self, x):
        # Input shape: (batch_size, 3, 200, 200)
        x = self.conv_layer(x)  # -> (batch_size, 32, 198, 198)
        x = self.relu(x)  # ReLU activation
        x = self.max_pool(x)  # -> (batch_size, 32, 99, 99)
        x = torch.flatten(x, 1)  # -> (batch_size, 32*99*99 = 313632)
        x = self.inner(x)  # -> (batch_size, 64)
        x = self.relu(x)  # ReLU activation
        x = self.output_layer(x)  # -> (batch_size, 1)

        return x
