import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


save_path = "./models"
loaded_model = tf.saved_model.load(save_path)


def create_img(image_path: str, output_path: str) -> Image:
    # 入力画像を読み込む
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)

    # 入力画像の前処理
    img = tf.image.resize(img, [256, 256])
    img = (img / 127.5) - 1
    img = tf.expand_dims(img, 0)

    # 戦闘顔の生成
    generated_image = loaded_model(img)

    # 生成された画像をピクセル値の範囲 [0, 1] に変換
    generated_image = (generated_image + 1) / 2.0

    # ピクセル値を [0, 255] の整数値にスケーリング
    generated_image = tf.cast(generated_image * 255, tf.uint8)

    # バッチの次元を削除
    generated_image = tf.squeeze(generated_image)

    # 生成された画像を numpy 配列に変換
    generated_image = generated_image.numpy()

    # OpenCVを用いて保存するために色チャネルをRGBからBGRへ変換
    generated_image = cv2.cvtColor(generated_image, cv2.COLOR_RGB2BGR)

    # createFightFace関数内での画像の保存（例）
    cv2.imwrite(output_path, generated_image)

    return generated_image


if __name__ == "__main__":
    create_img(image_path="./images/test.png", output_path="./images/result.png")
