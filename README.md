# attgan Implementation in Tensorflow
I am implementing AttgGAN just for study.

I referred the repository "<https://github.com/LynnHo/AttGAN-Tensorflow>"

Only changing hair color is a little bit done. (tf2)
![image](https://user-images.githubusercontent.com/26874750/81803306-74f08c80-9552-11ea-87fc-104b06ef8ce0.png)

## Usage
If you want to edit this project, locate CelebA data into

``attgan_impl_tf2/data/celeba/img_align_celeba/{ALL_IMAGES}``<br>
``attgan_impl_tf2/data/celeba/annotations/train_label.txt``<br>
``attgan_impl_tf2/data/celeba/annotations/val_label.txt``<br>
``attgan_impl_tf2/data/celeba/annotations/test_label.txt``<br>

CelebA dataset: [download link](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

And then, run ``data_spliter.ipynb`` in ``attgan_impl_tf2`` to split data into train/validation/test data.

If you want to train, check ``train_attgan.ipynb`` for single GPU, ``train_attgan_distributed.ipynb`` for multi-gpu. Some configuration is located in ``settings.py``, ``settings_distributed.py`` respectively.

Note that, this project is not perfect. This project is for studying!

**Thanks to this paper:**

He, Z., Zuo, W., Kan, M., Shan, S., & Chen, X. (2017). AttGAN: Facial Attribute Editing by Only Changing What You Want. *IEEE Transactions on Image Processing, 28*, 5464-5478.
