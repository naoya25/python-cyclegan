# pcのスペックによっては実行が終わりませんwwwwwww
# 10時間以上かかる場合もあります


import tensorflow as tf
import matplotlib.pyplot as plt
from examples.tensorflow_examples.models.pix2pix import pix2pix
import time
from IPython.display import clear_output
import pathlib

AUTOTUNE = tf.data.AUTOTUNE  # 入力パイプラインを最適化

BUFFER_SIZE = 1000
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
OUTPUT_CHANNELS = 3
LAMBDA = 10
EPOCHS = 1  # 俺のpcやと10時間かかった


# データのパスをここで指定します。
data_root = pathlib.Path("./dataset")
train_x_paths = list(data_root.glob("trainA/*"))
train_x_paths = [str(path) for path in train_x_paths][:100]

train_y_paths = list(data_root.glob("trainB/*"))
train_y_paths = [str(path) for path in train_y_paths][:100]


# arrayからtensorへ変換します
train_x = tf.data.Dataset.from_tensor_slices(train_x_paths)
train_y = tf.data.Dataset.from_tensor_slices(train_y_paths)

# 各データの出力数を変数に格納します
len_train_x = len(train_x)
len_train_y = len(train_y)


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


# 画像生成
generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type="instancenorm")
generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type="instancenorm")

discriminator_x = pix2pix.discriminator(norm_type="instancenorm", target=False)
discriminator_y = pix2pix.discriminator(norm_type="instancenorm", target=False)

# 損失関数
loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real, generated):
    real_loss = loss_obj(tf.ones_like(real), real)
    generated_loss = loss_obj(tf.zeros_like(generated), generated)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss * 0.5


def generator_loss(generated):
    return loss_obj(tf.ones_like(generated), generated)


def calc_cycle_loss(real_image, cycled_image):
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
    return LAMBDA * loss1


def identity_loss(real_image, same_image):
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return LAMBDA * 0.5 * loss


generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_path = "./checkpoints/train"

ckpt = tf.train.Checkpoint(
    generator_g=generator_g,
    generator_f=generator_f,
    discriminator_x=discriminator_x,
    discriminator_y=discriminator_y,
    generator_g_optimizer=generator_g_optimizer,
    generator_f_optimizer=generator_f_optimizer,
    discriminator_x_optimizer=discriminator_x_optimizer,
    discriminator_y_optimizer=discriminator_y_optimizer,
)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print("Latest checkpoint restored!!")


@tf.function
def train_step(real_x, real_y):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    with tf.GradientTape(persistent=True) as tape:
        # Generator G translates X -> Y
        # Generator F translates Y -> X.

        fake_y = generator_g(real_x, training=True)
        cycled_x = generator_f(fake_y, training=True)

        fake_x = generator_f(real_y, training=True)
        cycled_y = generator_g(fake_x, training=True)

        # same_x and same_y are used for identity loss.
        same_x = generator_f(real_x, training=True)
        same_y = generator_g(real_y, training=True)

        disc_real_x = discriminator_x(real_x, training=True)
        disc_real_y = discriminator_y(real_y, training=True)

        disc_fake_x = discriminator_x(fake_x, training=True)
        disc_fake_y = discriminator_y(fake_y, training=True)

        # calculate the loss
        gen_g_loss = generator_loss(disc_fake_y)
        gen_f_loss = generator_loss(disc_fake_x)

        total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)

        # Total generator loss = adversarial loss + cycle loss
        total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
        total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)

        disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
        disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

    # Calculate the gradients for generator and discriminator
    generator_g_gradients = tape.gradient(total_gen_g_loss, generator_g.trainable_variables)
    generator_f_gradients = tape.gradient(total_gen_f_loss, generator_f.trainable_variables)

    discriminator_x_gradients = tape.gradient(disc_x_loss, discriminator_x.trainable_variables)
    discriminator_y_gradients = tape.gradient(disc_y_loss, discriminator_y.trainable_variables)

    # Apply the gradients to the optimizer
    generator_g_optimizer.apply_gradients(
        zip(generator_g_gradients, generator_g.trainable_variables)
    )

    generator_f_optimizer.apply_gradients(
        zip(generator_f_gradients, generator_f.trainable_variables)
    )

    discriminator_x_optimizer.apply_gradients(
        zip(discriminator_x_gradients, discriminator_x.trainable_variables)
    )

    discriminator_y_optimizer.apply_gradients(
        zip(discriminator_y_gradients, discriminator_y.trainable_variables)
    )


for epoch in range(EPOCHS):
    start = time.time()

    n = 0
    for image_x, image_y in tf.data.Dataset.zip((train_x, train_y)):
        train_step(image_x, image_y)
        if n % 10 == 0:
            print(".", end="")
        n += 1

    clear_output(wait=True)

    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print("Saving checkpoint for epoch {} at {}".format(epoch + 1, ckpt_save_path))

    print("Time taken for epoch {} is {} sec\n".format(epoch + 1, time.time() - start))


# モデルの保存
ckpt_save_path = ckpt_manager.save()
generator_g.save("./models/saved_model")
print("Saving checkpoint and model at {}".format(ckpt_save_path))
