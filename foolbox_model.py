import torchvision.models as models
import foolbox


def create():
    squeezenet = models.squeezenet1_0(pretrained=True)
    fmodel = foolbox.models.PyTorchModel(
        squeezenet, (0, 1), num_classes=10)
    return fmodel