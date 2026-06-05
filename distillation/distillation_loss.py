import torch
import torch.nn.functional as F

def output_distillation_loss(student_outputs, teacher_outputs):
    """
    Computes the Mean Squared Error (MSE) between the student and teacher outputs.
    Both outputs are expected to be post-sigmoid weights in [0, 1].
    """
    return F.mse_loss(student_outputs, teacher_outputs)

def hint_distillation_loss(student_embs, teacher_embs, option='A'):
    """
    Computes the feature-level hint distillation loss based on the chosen option.
    
    Args:
        student_embs (list of torch.Tensor): Intermediate layer embeddings from the student.
        teacher_embs (list of torch.Tensor): Intermediate layer embeddings from the teacher.
        option (str): 'depth4_spaced', 'depth4_endpoints', or 'depth2_matched' indicating the mapping strategy.
            - 'depth4_spaced': Depth=4. Map L2->L1 and L4->L2
            - 'depth4_endpoints': Depth=4. Map L1->L1 and L4->L2
            - 'depth2_matched': Depth=2. Map L1->L1 and L2->L2
    """
    if option == 'depth4_spaced':
        # Student has 4 layers, Teacher has 2 layers.
        # Map L2 (idx 1) -> L1 (idx 0), and L4 (idx 3) -> L2 (idx 1)
        loss_l1 = F.mse_loss(student_embs[1], teacher_embs[0])
        loss_l2 = F.mse_loss(student_embs[3], teacher_embs[1])
        return loss_l1 + loss_l2

    elif option == 'depth4_endpoints':
        # Student has 4 layers, Teacher has 2 layers.
        # Map L1 (idx 0) -> L1 (idx 0), and L4 (idx 3) -> L2 (idx 1)
        loss_l1 = F.mse_loss(student_embs[0], teacher_embs[0])
        loss_l2 = F.mse_loss(student_embs[3], teacher_embs[1])
        return loss_l1 + loss_l2

    elif option == 'depth2_matched':
        # Student has 2 layers, Teacher has 2 layers.
        # Map L1 (idx 0) -> L1 (idx 0), and L2 (idx 1) -> L2 (idx 1)
        loss_l1 = F.mse_loss(student_embs[0], teacher_embs[0])
        loss_l2 = F.mse_loss(student_embs[1], teacher_embs[1])
        return loss_l1 + loss_l2

    else:
        raise ValueError(f"Invalid option: {option}. Choose 'depth4_spaced', 'depth4_endpoints', or 'depth2_matched'.")


def lsp_distillation_loss(student_final_emb, teacher_final_emb, batch, max_nodes_per_graph=200):
    """
    Local Structure Preservation (LSP) Distillation.
    Computes pairwise distance/similarity matrices for the final embeddings
    within each graph in the batch and penalizes their differences.
    
    Args:
        student_final_emb (torch.Tensor): Final node embeddings from the student.
        teacher_final_emb (torch.Tensor): Final node embeddings from the teacher.
        batch (torch.Tensor): Batch indices for each node, mapping node to graph.
        max_nodes_per_graph (int): Max number of nodes to sample per graph to prevent OOM.
    """
    batch_size = int(batch.max().item() + 1)
    total_loss = 0.0
    valid_graphs = 0
    
    for i in range(batch_size):
        # Extract nodes belonging to graph i
        mask = (batch == i)
        s_emb = student_final_emb[mask]
        t_emb = teacher_final_emb[mask]
        
        num_nodes = s_emb.shape[0]
        if num_nodes == 0:
            continue
            
        # Randomly sample nodes if graph is too large to save memory/compute
        if num_nodes > max_nodes_per_graph:
            indices = torch.randperm(num_nodes)[:max_nodes_per_graph]
            s_emb = s_emb[indices]
            t_emb = t_emb[indices]
            
        # Compute pairwise distance matrices (Cosine similarity is scale invariant and robust)
        # Normalize embeddings to compute cosine similarity easily via matrix multiplication
        s_emb_norm = F.normalize(s_emb, p=2, dim=1)
        t_emb_norm = F.normalize(t_emb, p=2, dim=1)
        
        s_sim_matrix = torch.matmul(s_emb_norm, s_emb_norm.T)
        t_sim_matrix = torch.matmul(t_emb_norm, t_emb_norm.T)
        
        # Compute MSE between the structural similarity matrices
        total_loss += F.mse_loss(s_sim_matrix, t_sim_matrix)
        valid_graphs += 1
        
    if valid_graphs == 0:
        return torch.tensor(0.0).to(student_final_emb.device)
        
    return total_loss / valid_graphs

def total_distillation_loss(student_outputs, teacher_outputs, student_embs, teacher_embs, batch,
                            alpha=1.0, beta=1.0, gamma=1.0, option='depth4_spaced'):
    """
    Combines the three distillation losses with given weights.
    
    Args:
        alpha (float): Weight for output distillation loss.
        beta (float): Weight for feature hint distillation loss.
        gamma (float): Weight for LSP distillation loss.
    """
    loss_out = output_distillation_loss(student_outputs, teacher_outputs)
    loss_hint = hint_distillation_loss(student_embs, teacher_embs, option=option)
    loss_lsp = lsp_distillation_loss(student_embs[-1], teacher_embs[-1], batch)
    
    total = (alpha * loss_out) + (beta * loss_hint) + (gamma * loss_lsp)
    return total, {'loss_out': loss_out.item(), 'loss_hint': loss_hint.item(), 'loss_lsp': loss_lsp.item()}
