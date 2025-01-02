import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class vitdino(nn.Module):
    def __init__(self, class_num=2, layers=2, img_size=224):
        super(vitdino, self).__init__()
        self.img_size = img_size
        
        print(f"number of layers frozen = {layers}")
        model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')        
        for i,child in enumerate(model.children()):
            if(i<2):
                for param in child.parameters():
                    param.requires_grad = False
            if(i==2):
                for j,child2 in enumerate(child.children()):
                    if(j<layers):
                        for param in child2.parameters():
                            param.requires_grad = False
                    else: 
                        for param in child2.parameters():
                            param.requires_grad = True
                        print(f'Unfroze layer {child2.__class__.__name__}')
            else: 
                for param in child.parameters():
                    param.requires_grad = False
        
        # self.model = model
        in_features = 384

        self.backbone = model

        self.multihead_attn = torch.nn.MultiheadAttention(embed_dim=in_features, num_heads=1, batch_first=True)

        self.mlp = nn.Sequential(
            nn.Linear(in_features, class_num), 
        )


    def forward(self, input_tensor):
        x = input_tensor.view(-1, 3, self.img_size, self.img_size)
        features = self.backbone(x)
        features = features.squeeze(-1).squeeze(-1)

        individual_tensors = features.view(input_tensor.shape[0], input_tensor.shape[1], -1)

        first_element = individual_tensors[:, 0, :]
        # query key value
        # import pdb; pdb.set_trace()
        attention_output, _ = self.multihead_attn(first_element.unsqueeze(1), individual_tensors, individual_tensors)
        output_tensor = attention_output.squeeze(1)

        output_tensor = self.mlp(output_tensor)

        return output_tensor, attention_output.squeeze(1)

if __name__=="__main__":
    model = vitdino()
    input_tensor = torch.randn((7, 5, 3, 224, 224))

    output = model(input_tensor)
    print("Output Shape:", output[1].shape)
    # import pdb; pdb.set_trace()
