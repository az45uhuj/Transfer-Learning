# reference https://medium.com/the-owl/extracting-features-from-an-intermediate-layer-of-a-pretrained-model-in-pytorch-c00589bda32b
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.io import read_image
from torch import nn
from torchsummary import summary

def show_layers(model0):
    Children_Counter = 0
    for i, j in model0.named_children():
        print("Children Counter:", Children_Counter, "Layer Name: ", i)
        Children_Counter += 1

def show_modules(model0):
    print(model0._modules)

class new_model(nn.Module):
    def __init__(self, output_layer = None, weights=None, model=None):
        super().__init__()
        self.weights = weights
        self.pretrained = model
        self.output_layer = output_layer
        self.layers = list(self.pretrained._modules.keys())
        self.layer_count = 0
        for l in self.layers:
            if l != self.output_layer:
                self.layer_count += 1
            else:
                break
        for i in range(1, len(self.layers)-self.layer_count):
            self.dummy_var = self.pretrained._modules.pop(self.layers[-i])

        self.net = nn.Sequential(self.pretrained._modules)

    def forward(self, x):
        x = self.net(x)
        return x

weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
mymodel = new_model(output_layer='layer4', weights=weights, model=model)
summary(mymodel, input_size=(3, 224, 224))

img = read_image("bird.jpg")
# step 2: Initialize the inference transforms
preprocess = weights.transforms()
# step 3: Apply inference preprocessing transforms
batch = preprocess(img).unsqueeze(0)
# step 4: Use the model and print the predicted category
extracted_feature = mymodel(batch)
print(extracted_feature)


#prediction = model(batch).squeeze(0).softmax(0)
#class_id = prediction.argmax().item()
#score = prediction[class_id].item()
#category_name = weights.meta["categories"][class_id]
#print(f"{category_name}:{100 * score:.1f}%")

