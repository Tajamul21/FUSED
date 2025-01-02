import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class vit_b_16(nn.Module):
    def __init__(self, class_num=2, layers_freeze=2, img_size=224):
        super(vit_b_16, self).__init__()
        self.img_size = img_size
        self.vit_b_16 = models.vit_b_16(weights='ViT_B_16_Weights.IMAGENET1K_V1')

        for param in self.vit_b_16.conv_proj.parameters():
            param.requires_grad = False
        for param in self.vit_b_16.encoder.parameters():
            param.requires_grad = False
            
        for i,child in enumerate(self.vit_b_16.encoder.layers.children()):
            # print(i, layers_freeze, child)
            if(i<layers_freeze):
                for param in child.parameters():
                    param.requires_grad = False
            else: 
                print(f'Unfroze layer {child.__class__.__name__}')
                for param in child.parameters():
                    param.requires_grad = True
                    

        self.vit_b_16.heads = nn.Sequential()
        # import pdb; pdb.set_trace()
        self.mlp = nn.Sequential(
            nn.Linear(768, class_num),
            nn.BatchNorm1d(class_num)
        )


    def forward(self, input_tensor):
        x = input_tensor.view(-1, 3, self.img_size, self.img_size)
        # import pdb; pdb.set_trace()
        features = self.vit_b_16(x)
        features = features.squeeze(-1).squeeze(-1)

        individual_tensors = features.view(input_tensor.shape[0], input_tensor.shape[1], -1)
        output_tensor1, _ = torch.max(individual_tensors, dim=1)
        output_tensor = self.mlp(output_tensor1)

        return output_tensor, output_tensor1

if __name__=="__main__":
    model = vit_b_16()
    input_tensor = torch.randn((7, 5, 3, 224, 224))

    output = model(input_tensor)
    print("Output Shape:", output[1].shape)

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