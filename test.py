# import pathlib
# import random
# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt

# # データのパスをここで指定します。
# data_root = pathlib.Path("./dataset")
# train_x_paths = list(data_root.glob("trainA/*"))
# train_x_paths = [str(path) for path in train_x_paths]

# train_y_paths = list(data_root.glob("trainB/*"))
# train_y_paths = [str(path) for path in train_y_paths]

# test_x_paths = list(data_root.glob("testA/*"))
# test_x_paths = [str(path) for path in test_x_paths]

# test_y_paths = list(data_root.glob("testB/*"))
# test_y_paths = [str(path) for path in test_y_paths]

# painting_path = random.choice(train_x_paths)
# photo_path = random.choice(train_y_paths)

# # 画像を読み込む
# painting = tf.keras.preprocessing.image.load_img(
#     painting_path,
# )
# photo = tf.keras.preprocessing.image.load_img(
#     photo_path,
# )
# # 読み込んだ画像をnumpy arrayに変換する
# painting = tf.keras.preprocessing.image.img_to_array(painting)
# photo = tf.keras.preprocessing.image.img_to_array(photo)

# # 画像の形状を(batch,h,w,channel)に拡張する(これがないとエラーが起こる)
# painting = painting[tf.newaxis] / 255
# photo = photo[tf.newaxis] / 255

# # 画像を表示する
# fig, axes = plt.subplots(1, 2, figsize=(10, 5))
# ax1 = axes[0]
# ax1.imshow(painting[0])
# ax1.set_xticks([])
# ax1.set_yticks([])
# ax1.set_title("Painting")

# ax2 = axes[1]
# ax2.imshow(photo[0])
# ax2.set_xticks([])
# ax2.set_yticks([])
# ax2.set_title("Photograph")

# plt.show()


import pathlib
import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

AUTOTUNE = tf.data.AUTOTUNE

# データのパスをここで指定します。
data_root = pathlib.Path("./dataset")
train_x_paths = list(data_root.glob("trainA/*"))
train_x_paths = [str(path) for path in train_x_paths][:100]

train_y_paths = list(data_root.glob("trainB/*"))
train_y_paths = [str(path) for path in train_y_paths][:100]

test_x_paths = list(data_root.glob("testA/*"))
test_x_paths = [str(path) for path in test_x_paths][:100]

test_y_paths = list(data_root.glob("testB/*"))
test_y_paths = [str(path) for path in test_y_paths][:100]


# 訓練用の写真をクロッピング、左右反転、正規化するための関数
def preprocess_image_train(path):
    image = tf.io.read_file(path)
    # 生データのテンソルを画像のテンソルに変換する。
    # これによりshape=(240,240,3)になる
    image = tf.image.decode_jpeg(image, channels=3)
    # モデルに合わせてリサイズする
    image = tf.image.resize(image, [286, 286], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # 指定した範囲を切り取るクロッピング
    image = tf.image.random_crop(image, size=[256, 256, 3])

    # ランダムなミラーリング
    image = tf.image.random_flip_left_right(image)
    # 正規化を行う
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1

    return image


def preprocess_image_test(path):
    image = tf.io.read_file(path)
    # 生データのテンソルを画像のテンソルに変換する。
    # これによりshape=(240,240,3)になる
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.random_crop(image, size=[256, 256, 3])

    # ランダムなミラーリング
    image = tf.image.random_flip_left_right(image)
    # 正規化を行う
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1

    return image


# arrayからtensorへ変換します
train_x = tf.data.Dataset.from_tensor_slices(train_x_paths)
train_y = tf.data.Dataset.from_tensor_slices(train_y_paths)
test_x = tf.data.Dataset.from_tensor_slices(test_x_paths)
test_y = tf.data.Dataset.from_tensor_slices(test_y_paths)

# 各データの出力数を変数に格納します
len_train_x = len(train_x)
len_train_y = len(train_y)
len_test_x = len(test_x)
len_test_y = len(test_y)

# 画像のパスから画像データをtensorとして取り出し、前処理、バッチ化を行います。
train_x = (
    train_x.map(preprocess_image_train, num_parallel_calls=AUTOTUNE)
    .cache()
    .shuffle(len_train_x)
    .batch(1)
)

train_y = (
    train_y.map(preprocess_image_train, num_parallel_calls=AUTOTUNE)
    .cache()
    .shuffle(len_train_y)
    .batch(1)
)

test_x = (
    test_x.map(preprocess_image_test, num_parallel_calls=AUTOTUNE)
    .cache()
    .shuffle(len_test_x)
    .batch(1)
)

test_y = (
    test_y.map(preprocess_image_test, num_parallel_calls=AUTOTUNE)
    .cache()
    .shuffle(len_test_y)
    .batch(1)
)
