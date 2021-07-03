
# embeddings
Wikipedia Dataset
train set: 2173   test set: 462   val set: 231   categories: 10

The features of images are extracted with VGG19.
The features of images are extracted with BERT.

train_img.mat -> image features in train set, dimension [2173, 4096].
train_txt.mat -> text features in train set, dimension [2173, 768].
train_img_lab.mat -> labels in train set, dimension [2173, 10].

test_img.mat -> image features in test set, dimension [492, 4096].
test_txt.mat -> text features in test set, dimension [492, 768].
test_img_lab.mat -> labels in test set, dimension [492, 10].     

