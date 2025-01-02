import torch
import torch.nn as nn
import clip


class CLIP_IMAGE(nn.Module):
    def __init__(self, image_layers_freeze, class_num):
        super(CLIP_IMAGE, self).__init__()
        model, _ = clip.load("ViT-B/16", jit=False)
        self.model = model
        self.image_model = self.model.visual

        for k in self.model.transformer.parameters():  
            k.requires_grad=False
        
        for i, child in enumerate(self.image_model.children()):
            if (i==2):
                for index, chldtwo in enumerate(child.resblocks):
                    if index < image_layers_freeze:
                        for param in chldtwo.parameters():
                            param.requires_grad = False
                    else: 
                        for param in chldtwo.parameters():
                            param.requires_grad = True
                        print(f"Unfroze Layer #{index+1}/{chldtwo.__class__.__name__} of Vision Transformer")
            elif (i==3):
                for param in chldtwo.parameters():
                    param.requires_grad = True
                print(f"Unfroze Layer Normalisation in last layer of Vision Transformer")
            else: 
                for param in child.parameters():
                        param.requires_grad = False
        self.mlp = nn.Sequential(
            nn.Linear(512, class_num),
            nn.BatchNorm1d(class_num)
        )

    def forward(self, input_tensor):
        x = input_tensor.view(-1, 3, 224, 224)
        features = self.model.encode_image(x)
        features = features.squeeze(-1).squeeze(-1)

        individual_tensors = features.view(input_tensor.shape[0], input_tensor.shape[1], -1)
        output_tensor, _ = torch.max(individual_tensors, dim=1)
        output_tensor1 = output_tensor.to(torch.float32)
        output_tensor = self.mlp(output_tensor1)

        return output_tensor, output_tensor1
    
    def convert_weights(self):
        """Convert applicable model parameters to fp16"""

        def _convert_weights_to_fp16(l):
            if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                l.weight.data = l.weight.data.half()
                if l.bias is not None:
                    l.bias.data = l.bias.data.half()

            if isinstance(l, nn.MultiheadAttention):
                for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                    tensor = getattr(l, attr)
                    if tensor is not None:
                        tensor.data = tensor.data.half()

            for name in ["text_projection", "proj"]:
                if hasattr(l, name):
                    attr = getattr(l, name)
                    if attr is not None:
                        attr.data = attr.data.half()

        self.model.apply(_convert_weights_to_fp16)

    def convert_models_to_fp32(self): 
        for p in self.model.parameters(): 
            if(p!=None and p.grad!=None):
                p.data = p.data.float() 
                p.grad.data = p.grad.data.float()
    
if __name__=="__main__":
    device = 'cuda:0'
    model = CLIP_IMAGE(6)

    input_tensor = torch.randn((7, 5, 3, 224, 224))
    input_tensor = input_tensor.to(device)
    model = model.to(device)

    output = model(input_tensor)
    print("Output Shape:", output.shape)