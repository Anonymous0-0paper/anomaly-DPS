import torch
import torch.nn as nn
import numpy as np
import os
import time
from .metrics import evaluate_model, plot_training_history

def train_model(model, train_loader, val_loader, num_classes, output_dir, device, epochs=10, lr=0.001):
    """Train model"""
    print("\n" + "=" * 80)
    print(f"{'Start Model Training':^80}")
    print("=" * 80)

    print(f"Using device: {device}")
    model = model.to(device)

    # Calculate class weights (for handling imbalanced data)
    all_labels = []
    for batch in train_loader:
        all_labels.extend(batch['y'].numpy())

    class_counts = np.bincount(all_labels, minlength=num_classes)
    class_weights = 1.0 / np.sqrt(class_counts + 1e-5)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Create learning rate scheduler (remove verbose parameter)
    try:
        # Try to create a scheduler with verbose parameter
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )
    except TypeError:
        # If it fails, create a scheduler without verbose parameter
        print("Warning: ReduceLROnPlateau does not support verbose parameter, using silent mode")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3
        )

    best_val_f1 = 0.0
    history = {'train_loss': [], 'val_acc': [], 'val_f1': []}

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()

        for batch in train_loader:
            x, y = batch['x'].to(device), batch['y'].to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            # Calculate training accuracy
            _, predicted = torch.max(logits.data, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

            train_loss += loss.item() * y.size(0)

        train_loss /= len(train_loader.dataset)
        train_acc = correct / total

        # Validation phase
        val_metrics = evaluate_model(model, val_loader, device, num_classes)
        val_acc = val_metrics['accuracy']
        val_f1 = val_metrics['f1']

        # Update learning rate
        scheduler.step(val_f1)

        # Record history
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        # Calculate epoch time
        epoch_time = time.time() - start_time

        print(f"\nEpoch {epoch + 1}/{epochs} ({epoch_time:.1f}s):")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        print(f"  Current learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Save the best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model_path = os.path.join(output_dir, "best_model.pth")
            torch.save(model.state_dict(), model_path)
            print(f"  🎯 Saving new best model to {model_path}, Validation F1: {val_f1:.4f}")

    print(f"\nTraining completed. Best validation F1 score: {best_val_f1:.4f}")

    # Save training history
    history_path = os.path.join(output_dir, "training_history.png")
    plot_training_history(history, history_path)

    return model, history
