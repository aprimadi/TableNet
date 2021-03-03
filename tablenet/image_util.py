import tensorflow as tf
import matplotlib.pyplot as plt

IMG_WIDTH = IMG_HEIGHT = 256

def normalize(image):
    image = tf.cast(image, tf.float32) / 255.0
    return image

def decode_jpeg(img, channels=3):
    img = tf.image.decode_jpeg(img, channels=channels)
    return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

def display_image_and_masks(image, table_mask, column_mask):
    plt.figure(figsize=(15, 15))

    display_list = [image, table_mask, column_mask]
    title = ['Image', 'Table Mask', 'Column Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

def display_image(image):
    plt.figure(figsize=(15, 15))
    plt.imshow(tf.keras.preprocessing.image.array_to_img(image))
    plt.show()

