"""Small CPU sequence pilot; richer architectures require demonstrated benefit."""
import torch
class ShapeGRU(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.gru=torch.nn.GRU(1,16,batch_first=True); self.output=torch.nn.Linear(16,1)
    def forward(self,x):
        _,hidden=self.gru(x); return self.output(hidden[-1]).squeeze(-1)
