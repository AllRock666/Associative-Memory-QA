import torch
import torch.nn as nn
from core.hopfield_torch import HopfieldRetriever

class HopfieldVQAModel(nn.Module):
    """
    Heterogeneous Text+Image Hopfield Memory for VQA.
    Extracts patch embeddings (ViT) and token embeddings (Text),
    concatenates them into a joint memory matrix, and retrieves
    the answer state using Continuous Hopfield attractor dynamics.
    """
    def __init__(self, clip_model_name: str = "openai/clip-vit-base-patch32", num_answers: int = 1000):
        super().__init__()
        try:
            from transformers import CLIPModel
        except ImportError:
            raise ImportError("Please install transformers.")
            
        # Load CLIP components
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        
        # Freeze CLIP for this demonstration (since we just want to test the Hopfield head)
        for param in self.clip.parameters():
            param.requires_grad = False
            
        hidden_dim = self.clip.config.projection_dim
        
        # We need a learnable query token [TASK] to prompt the Hopfield network
        self.task_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # The core Hopfield retriever
        self.hopfield = HopfieldRetriever(beta=1.0, n_iter=1)
        
        # Answer classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, num_answers)
        )

    def forward(self, input_ids, attention_mask, pixel_values):
        """
        Args:
            input_ids: (B, L)
            attention_mask: (B, L)
            pixel_values: (B, C, H, W)
            
        Returns:
            logits: (B, num_answers)
            attention_weights: (B, 1, N + L) - The Hopfield attention over patches and tokens
        """
        B = input_ids.size(0)
        
        # 1. Extract Visual Patch Embeddings
        # Vision output: last_hidden_state is (B, 50, 768) for patch32
        vision_outputs = self.clip.vision_model(pixel_values=pixel_values)
        # Project to shared space (B, 50, 512)
        visual_memory = self.clip.visual_projection(vision_outputs.last_hidden_state)
        
        # 2. Extract Text Token Embeddings
        # Text output: last_hidden_state is (B, L, 512)
        text_outputs = self.clip.text_model(input_ids=input_ids, attention_mask=attention_mask)
        # Project to shared space (B, L, 512)
        text_memory = self.clip.text_projection(text_outputs.last_hidden_state)
        
        # 3. Build Heterogeneous Memory Matrix X = [V; T]
        # Shape: (B, 50 + L, 512)
        joint_memory = torch.cat([visual_memory, text_memory], dim=1)
        
        # 4. Expand task token for batch
        # Shape: (B, 1, 512)
        query = self.task_token.expand(B, -1, -1)
        
        # 5. Continuous Hopfield Retrieval
        # Retrieve the joint multimodal state
        updated_state, attention_weights, _ = self.hopfield(query, joint_memory, top_k=5)
        
        # 6. Predict Answer
        # updated_state is (B, 1, 512) -> squeeze to (B, 512)
        logits = self.classifier(updated_state.squeeze(1))
        
        return logits, attention_weights
