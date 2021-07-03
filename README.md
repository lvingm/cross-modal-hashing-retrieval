
# 1. Embeddings of Wikipedia Dataset
- train set: 2173   test set: 462   val set: 231   categories: 10
- The features of images are extracted with VGG19.
- The features of images are extracted with BERT.

# 2. Important file
- train_self.py: to train self-net
- train_others.py: to train others-net
- train_nso.py: to train simple model without self-others net

# 3. pipeline
- python vgg_bow_mat.py  # to extract embedding and save as .mat file
- python few_shot_dataset.py  # to split dataset into few-shot and normal
- python train_self.py  # to train self-net
- python train_others.py  # to train others-net
