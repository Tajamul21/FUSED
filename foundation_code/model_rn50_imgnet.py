import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class resnet(nn.Module):
    def __init__(self, layers=2, img_size=224):
        super(resnet, self).__init__()
        self.img_size = img_size
        resnet50 = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V1')

        # layers = 2
        for i, child in enumerate(resnet50.children()):
            if(i<layers+2):
                for param in child.parameters():
                    param.requires_grad = False
            else: 
                for param in child.parameters():
                    param.requires_grad = True
                print(f'Unfroze layer {child.__class__.__name__}')

        self.backbone = nn.Sequential(*list(resnet50.children())[:-1])
        self.mlp = nn.Sequential(
            nn.BatchNorm1d(resnet50.fc.in_features),
            nn.Linear(resnet50.fc.in_features, 2),
            nn.BatchNorm1d(2)
        )


    def forward(self, input_tensor):
        x = input_tensor.view(-1, 3, self.img_size, self.img_size)
        features = self.backbone(x)
        features = features.squeeze(-1).squeeze(-1)

        individual_tensors = features.view(input_tensor.shape[0], input_tensor.shape[1], -1)
        output_tensor, _ = torch.max(individual_tensors, dim=1)
        output_tensor = self.mlp(output_tensor)

        return output_tensor

if __name__=="__main__":
    model = resnet()
    input_tensor = torch.randn((7, 5, 3, 224, 224))

    output = model(input_tensor)
    print("Output Shape:", output.shape)

# (batch_size, elements, 3, 224, 224) - torch.Size([7, 5, 3, 224, 224])
# (batch_size * elements, 3, 224, 224) - torch.Size([35, 3, 224, 224])
# (batch_size * elements, 2048) - torch.Size([35, 2048])
# (batch_size, elements, 2048) - torch.Size([7, 5, 2048])
        # ---------------------------------
# Max Pooling
# output_tensor = torch.max(individual_tensors, dim=1)
# (batch_size, features) - torch.Size([7, 2048])

# Attention
    # (batch_size, features) - torch.Size([7, 2048])
# ---------------------------------