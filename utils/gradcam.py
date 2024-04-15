import torch
from torch.autograd import Function


class GradCAM:
    def __init__(self, model, layer_name):
        self.model = model
        self.layer_name = layer_name
        self.model.eval()
        self.feature_maps = None
        self.gradients = None
        self.register_hooks()

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_hook(self, module, input, output):
        self.feature_maps = output

    def register_hooks(self):
        for name, module in self.model.named_modules():
            if name == self.layer_name:
                module.register_forward_hook(self.forward_hook)
                module.register_backward_hook(
                    lambda module, grad_in, grad_out: self.save_gradient(grad_out[0])
                )

    def __call__(self, x, index=None):
        output = self.model(x)
        if index is None:
            index = output.argmax(dim=1).item()

        one_hot = torch.zeros_like(output)
        one_hot[0, index] = 1
        self.model.zero_grad()
        output.backward(gradient=one_hot, retain_graph=True)

        gradients = self.gradients
        feature_maps = self.feature_maps
        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)

        grad_cam = torch.sum(weights * feature_maps, dim=1).squeeze(0)
        grad_cam = torch.clamp(grad_cam, min=0)

        return grad_cam
