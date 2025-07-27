import timm

def get_resnet(model_name='resnet50', pretrained=True):
    model = timm.create_model(model_name, pretrained=pretrained)
    return model 