import torch
import torch.nn as nn
import clip

class CLIP_IMAGE(nn.Module):
    def __init__(self, image_layers_freeze):
        super(CLIP_IMAGE, self).__init__()
        model, _ = clip.load("RN50", device="cuda", jit=False)
        self.model = model
        self.image_model = self.model.visual

        # Freezing the other half 
        for k in self.model.transformer.parameters():  
            k.requires_grad=False
        #
        # import pdb; pdb.set_trace()
        if(image_layers_freeze==0):        
            for k in self.image_model.parameters():
                k.requires_grad=True
        else:
            for i, child in enumerate(self.image_model.children()):
                if (i<11):
                    for param in child.parameters():
                            param.requires_grad = False
                else: 
                    if(i-10<image_layers_freeze-1):
                        for child2 in child.children():
                            for param in child2.parameters():
                                param.requires_grad = False
                    else:
                        for child2 in child.children():
                            for param in child2.parameters():
                                param.requires_grad = True
                                # print(param.requires_grad)
                        print(f"Unfroze Layer #{child2.__class__.__name__} of Vision Resnet")

        self.mlp = nn.Sequential(
            nn.Linear(1024, 2),
            nn.BatchNorm1d(2)
        )

    def forward(self, input_tensor):
        x = input_tensor.view(-1, 3, 224, 224)
        features = self.model.encode_image(x)
        features = features.squeeze(-1).squeeze(-1)

        individual_tensors = features.view(input_tensor.shape[0], input_tensor.shape[1], -1)
        output_tensor, _ = torch.max(individual_tensors, dim=1)
        output_tensor = output_tensor.to(torch.float32)
        output_tensor = self.mlp(output_tensor)

        return output_tensor
    
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
    model = CLIP_IMAGE(2)

    input_tensor = torch.randn((7, 5, 3, 224, 224))
    input_tensor = input_tensor.to('cuda')
    model = model.to('cuda')

    output = model(input_tensor)
    print("Output Shape:", output.shape)