import tensorflow.compat.v1 as tf
import numpy as np
import yaml
import os
import csv
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image
from Train_final_combined import create_downscaler, create_upscaler


tf.disable_v2_behavior()

SUPPORTED_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.bmp']

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_image(image_path, mode='RGB'):
    image = Image.open(image_path).convert(mode)
    return np.array(image) / 255.0

def save_image(image_array, path):
    image = (image_array * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(image).save(path)

def load_model(sess, checkpoint_path, scope):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        if not variables:
            raise ValueError(f"No variables found in scope '{scope}'. Ensure model is defined before restoring.")
        saver = tf.train.Saver(var_list=variables)
        saver.restore(sess, checkpoint_path)

def evaluate_metrics(original, reconstructed):
    psnr_value = psnr(original, reconstructed, data_range=1.0)
    ssim_value = ssim(original, reconstructed, multichannel=True, data_range=1.0)
    return psnr_value, ssim_value

def inference(config_path):
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)

    image_path = config['image']
    model_type = config['model_type']
    checkpoint = config['checkpoint']
    evaluate = config['evaluate']
    model_mod = config['model_mod']

    is_folder = os.path.isdir(image_path)
    image_list = []

    if is_folder:
        image_list = [os.path.join(image_path, f) for f in os.listdir(image_path)
                      if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS]
    else:
        image_list = [image_path]

    metrics_log = []

    with tf.Session() as sess:
        if model_type in ['downscaler', 'both']:
            downscaler_inputs, downscaler_outputs = create_downscaler(1, model_mod)
            if model_mod == 'IRN':
                load_model(sess, checkpoint.replace("upscaler", "downscaler"), "downscaler")

        if model_type in ['upscaler', 'both']:
            upscaler_inputs, upscaler_outputs = create_upscaler(1, model_mod)
            load_model(sess, checkpoint, "upscaler")

        for img_path in image_list:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            image = load_image(img_path, mode='RGB')
            image_input = np.expand_dims(image, axis=0)

            if model_type in ['downscaler', 'both']:
                downscaled = sess.run(downscaler_outputs, feed_dict={downscaler_inputs: image_input})
                downscaled = np.squeeze(downscaled)

                if model_type == 'downscaler':
                    downscaled_3ch = np.mean(downscaled, axis=-1, keepdims=True).repeat(3, axis=-1)
                    save_image(downscaled_3ch, os.path.join(OUTPUT_DIR, f"{base_name}_downscaler.png"))

            if model_type in ['upscaler', 'both']:
                if model_type == 'both':
                    upscale_input = np.expand_dims(downscaled, axis=0)
                elif model_mod == 'IRN':
                    image = load_image(img_path, mode='RGBA')
                    upscale_input = np.expand_dims(image, axis=0)
                else:
                    upscale_input = image_input

                upscaled = sess.run(upscaler_outputs, feed_dict={upscaler_inputs: upscale_input})
                upscaled = np.squeeze(upscaled)

                suffix = "reconstructed" if model_type == 'both' else "upscaler"
                save_image(upscaled, os.path.join(OUTPUT_DIR, f"{base_name}_{suffix}.png"))

                if evaluate and model_type == 'both':
                    original = load_image(img_path, mode='RGB')
                    psnr_val, ssim_val = evaluate_metrics(original, upscaled)
                    metrics_log.append([base_name, psnr_val, ssim_val])

    if evaluate and model_type == 'both':
        with open(os.path.join(OUTPUT_DIR, 'metrics.csv'), mode='w', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(['filename', 'PSNR', 'SSIM'])
            writer.writerows(metrics_log)

if __name__ == "__main__":
    inference("infer_config.yml")