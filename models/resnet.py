import timm

def get_resnet(model_name='resnet50', pretrained=True):
    model = timm.create_model(model_name, pretrained=pretrained)
    return model

resnet50 = lambda pretrained=True: get_resnet('resnet50', pretrained)
resnet18 = lambda pretrained=True: get_resnet('resnet18', pretrained)