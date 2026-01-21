import torch


# --- 3. Training Function ---
def train_epoch(model, optimizer, device, data_loader):
    """Training for one epoch"""
    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    
    for iter, batch in enumerate(data_loader):
        # Move entire batch to device at once
        batch = batch.to(device)
        batch_labels = batch.y
        batch_x = batch.x
        batch_e = batch.edge_attr if batch.edge_attr is not None else None
        
        optimizer.zero_grad()
        
        # Forward pass
        batch_scores = model.forward(batch, batch_x, batch_e)
        loss = model.loss(batch_scores, batch_labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Metrics
        epoch_loss += loss.detach().item()
        pred = batch_scores.max(dim=1)[1]
        epoch_train_acc += pred.eq(batch_labels).sum().item()
        nb_data += batch_labels.size(0)
    
    epoch_loss /= (iter + 1)
    epoch_train_acc /= nb_data
    
    return epoch_loss, epoch_train_acc


# --- 4. Evaluation Function ---
def evaluate_network(model, device, data_loader):
    """Evaluation"""
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    
    with torch.no_grad():
        for iter, batch in enumerate(data_loader):
            # Move entire batch to device at once
            batch = batch.to(device)
            batch_labels = batch.y
            batch_x = batch.x
            batch_e = batch.edge_attr if batch.edge_attr is not None else None
            
            batch_scores = model.forward(batch, batch_x, batch_e)
            loss = model.loss(batch_scores, batch_labels)
            
            epoch_test_loss += loss.detach().item()
            pred = batch_scores.max(dim=1)[1]
            epoch_test_acc += pred.eq(batch_labels).sum().item()
            nb_data += batch_labels.size(0)
        
        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= nb_data
    
    return epoch_test_loss, epoch_test_acc