import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class ensemble(nn.Module):
    def __init__(self, layers=2, layers_freeze=2, img_size=224):
        super(ensemble, self).__init__()
        self.img_size = img_size
        
        print(f"number of layers frozen = {layers}")
        model_vitdino = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')        
        for i,child in enumerate(model_vitdino.children()):
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

        self.backbone_vitdino = model_vitdino

        self.multihead_attn = torch.nn.MultiheadAttention(embed_dim=in_features, num_heads=1, batch_first=True)

        # VIT 16 IMAGENET
        
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

        in_features_mlp = 1152
        self.mlp = nn.Sequential(
            nn.Linear(in_features_mlp, 2), 
            nn.BatchNorm1d(2)
        )


    def forward(self, input_tensor):
        
        # VITDINO
        
        x = input_tensor.view(-1, 3, self.img_size, self.img_size)
        features_vitdino = self.backbone_vitdino(x)
        features_vitdino = features_vitdino.squeeze(-1).squeeze(-1)

        individual_tensors_vitdino = features_vitdino.view(input_tensor.shape[0], input_tensor.shape[1], -1)

        first_element_vitdino = individual_tensors_vitdino[:, 0, :]
        # query key value
        # import pdb; pdb.set_trace()
        attention_output_vitdino, _ = self.multihead_attn(first_element_vitdino.unsqueeze(1), individual_tensors_vitdino, individual_tensors_vitdino)
        features_vitdino = attention_output_vitdino.squeeze(1)

        
        
        
        # IMAGENET
        
        features_imagenet = self.vit_b_16(x)
        features_imagenet = features_imagenet.squeeze(-1).squeeze(-1)

        individual_tensors_imagenet = features_imagenet.view(input_tensor.shape[0], input_tensor.shape[1], -1)
        features_imagenet, _ = torch.max(individual_tensors_imagenet, dim=1)
        
        # ENSEMBLE MODEL
        # import pdb; pdb.set_trace()
        
        combined_features = torch.cat((features_vitdino, features_imagenet), dim=1)
        
        
        output_tensor_ensemble = self.mlp(combined_features)
        
        
        

        return output_tensor_ensemble, combined_features

if __name__=="__main__":
    model = ensemble()
    input_tensor = torch.randn((7, 5, 3, 224, 224))

    output = model(input_tensor)
    print("Output Shape:", output[1].shape)
