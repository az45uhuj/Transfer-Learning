# Transfer-Learning
# https://cs231n.github.io/transfer-learning/

"When and how to fine-tune? How do you decide what type of transfer learning you should perform on a new dataset? This is a function of several factors, but the two most important ones are the size of the new dataset (small or big), and its similarity to the original dataset (e.g. ImageNet-like in terms of the content of images and the classes, or very different, such as microscope images). Keeping in mind that ConvNet features are more generic in early layers and more original-dataset-specific in later layers, here are some common rules of thumb for navigating the 4 major scenarios:

New dataset is small and similar to original dataset. Since the data is small, it is not a good idea to fine-tune the ConvNet due to overfitting concerns.
Since the data is similar to the original data, we expect higher-level features in the ConvNet to be relevant to this dataset as well. Hence, the best idea
might be to train a linear classifier on the CNN codes.

New dataset is large and similar to the original dataset. Since we have more data, we can have more confidence that we won’t overfit if we were to try to 
fine-tune through the full network.

New dataset is small but very different from the original dataset. Since the data is small, it is likely best to only train a linear classifier. Since the
dataset is very different, it might not be best to train the classifier form the top of the network, which contains more dataset-specific features. 
Instead, it might work better to train the SVM classifier from activations somewhere earlier in the network.

New dataset is large and very different from the original dataset. Since the dataset is very large, we may expect that we can afford to train a ConvNet
from scratch. However, in practice it is very often still beneficial to initialize with weights from a pretrained model. In this case, we would have enough
data and confidence to fine-tune through the entire network."

