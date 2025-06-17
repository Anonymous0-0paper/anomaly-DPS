import torch
import torch.nn as nn
import numpy as np
import os
import time
import gc
from tqdm import tqdm
from .metrics import evaluate_model, plot_training_history

def train_model(model, train_loader, val_loader, num_classes, output_dir, device,
                epochs=10, lr=0.001, weight_decay=1e-4, max_grad_norm=1.0):
    """Unified training function supporting supervised and unsupervised modes"""
    print("\n" + "=" * 80)
    print(f"{'Start Unified Model Training':^80}")
    print("=" * 80)

    print(f"Using device: {device}")
    model = model.to(device)

    # Detect training mode
    training_mode = getattr(train_loader.dataset, 'training_mode', 'supervised')
    print(f"Training mode: {'Unsupervised' if training_mode == 'unsupervised' else 'Supervised'}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "best_model.pth")

    # ===== 1. Optimize class distribution calculation (supervised mode only) =====
    if training_mode == 'supervised':
        print("\n[Optimized] Computing class distribution...")
        start_time = time.time()

        # Use precomputed class distribution
        if hasattr(train_loader.dataset, 'class_counts'):
            class_counts = train_loader.dataset.class_counts
            print(f"  Precomputed class distribution: {class_counts} (in {time.time() - start_time:.2f}s)")
        # Efficiently compute class distribution
        else:
            class_counts = _compute_class_counts_fast(train_loader, num_classes)
            print(f"  Computed class distribution: {class_counts} (in {time.time() - start_time:.2f}s)")

        # Calculate class weights
        class_weights = 1.0 / np.sqrt(class_counts + 1)
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        print(f"Class weights: {class_weights.cpu().numpy().round(3)}")

    # ===== 2. Loss function and optimizer =====
    if training_mode == 'supervised':
        criterion = nn.CrossEntropyLoss(weight=class_weights if 'class_weights' in locals() else None)
    else:
        criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr / 100
    )

    # ===== 3. Training state tracking =====
    best_val_f1 = 0.0
    history = {'train_loss': [], 'val_acc': [], 'val_f1': []}
    early_stop_counter = 0
    PATIENCE = 5

    # Device-specific settings
    use_amp = (device.type == 'cuda')  # Only CUDA supports mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ===== 4. Training loop =====
    print("\nStarting training...")
    for epoch in range(epochs):
        try:
            # ===== Training phase =====
            model.train()
            epoch_loss = 0.0
            start_time = time.time()

            # Set memory cleanup interval
            CLEAN_CACHE_INTERVAL = 100 if device.type in ['cuda', 'mps'] else float('inf')

            batch_iterator = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}",
                                  total=len(train_loader), unit="batch", dynamic_ncols=True)

            for batch_idx, batch in enumerate(batch_iterator):
                x = batch['x'].to(device, non_blocking=True)

                # Clear gradients
                optimizer.zero_grad()

                # Mixed precision training
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    if training_mode == 'supervised':
                        # Supervised mode
                        y = batch['y'].to(device, non_blocking=True)
                        logits, _ = model(x, training_mode='supervised')
                        loss = criterion(logits, y)
                    else:
                        # Unsupervised mode
                        target = batch['x'].to(device, non_blocking=True)  # Reconstruction target is the input itself
                        _, reconstructed = model(x, training_mode='unsupervised')
                        loss = criterion(reconstructed, target)

                # Backward propagation
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()

                # Update statistics
                epoch_loss += loss.item() * x.size(0)

                # Periodically clean memory
                if (batch_idx + 1) % CLEAN_CACHE_INTERVAL == 0:
                    if device.type == 'mps':
                        torch.mps.empty_cache()
                    elif device.type == 'cuda':
                        torch.cuda.empty_cache()
                    gc.collect()

                # Update progress bar
                batch_iterator.set_postfix(loss=loss.item())

            # Calculate average training loss
            train_loss = epoch_loss / len(train_loader.dataset)

            # ===== Validation phase =====
            print(f"\nValidating epoch {epoch + 1}...")
            val_metrics = evaluate_model(model, val_loader, device, num_classes, training_mode)
            epoch_time = time.time() - start_time

            # Record history
            history['train_loss'].append(train_loss)
            history['val_acc'].append(val_metrics['accuracy'])
            history['val_f1'].append(val_metrics['f1'])

            # Print training information
            print(f"\nEpoch {epoch + 1}/{epochs} ({epoch_time:.1f}s): "
                  f"Train Loss: {train_loss:.4f} | Val F1: {val_metrics['f1']:.4f} "
                  f"| LR: {optimizer.param_groups[0]['lr']:.2e}")

            # Update learning rate
            scheduler.step()

            # Early stopping mechanism and model saving
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                torch.save(model.state_dict(), model_path)
                print(f"Saved best model (F1: {best_val_f1:.4f})")
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= PATIENCE:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        except Exception as e:
            print(f"\nError in epoch {epoch + 1}: {str(e)}")
            print("Saving emergency model...")
            emergency_path = os.path.join(output_dir, f"emergency_model_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), emergency_path)
            print(f"Saved emergency model to {emergency_path}")
            break

    # Load the best model
    try:
        model.load_state_dict(torch.load(model_path))
        print(f"\nTraining completed. Best validation F1: {best_val_f1:.4f}")
    except:
        print("\nCould not load best model, using final model weights")

    # Save training history
    try:
        history_path = os.path.join(output_dir, "training_history.png")
        plot_training_history(history, history_path)
    except Exception as e:
        print(f"Failed to save training history: {str(e)}")

    return model, history

def _compute_class_counts_fast(loader, num_classes):
    """Efficiently compute class distribution"""
    counts = np.zeros(num_classes, dtype=int)

    # Use batch processing for speed
    for batch in loader:
        labels = batch['y'].numpy()
        batch_counts = np.bincount(labels, minlength=num_classes)
        counts += batch_counts

    return counts

def _estimate_class_counts(loader, num_classes, sample_fraction=0.1):
    """Estimate class distribution by sampling"""
    counts = np.zeros(num_classes, dtype=int)
    total_batches = max(1, int(len(loader) * sample_fraction))

    for i, batch in enumerate(loader):
        if i >= total_batches:
            break
        labels = batch['y'].numpy()
        batch_counts = np.bincount(labels, minlength=num_classes)
        counts += batch_counts

    # Scale up the estimate proportionally
    if len(loader) > 0:
        scale_factor = len(loader) / total_batches
        counts = (counts * scale_factor).astype(int)

    return counts
