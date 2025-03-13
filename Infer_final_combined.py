import tensorflow.compat.v1 as tf
import numpy as np
import yaml
import os
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image
from Train_final_combined import create_downscaler, create_upscaler

tf.disable_v2_behavior()

def load_image(image_path, mode='RGB'):
    image = Image.open(image_path).convert(mode)
    return np.array(image) / 255.0

def save_image(image_array, path):
    image = (image_array * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(image).save(path)

def load_model(sess, checkpoint_path, scope):
    """ Load model from checkpoint ensuring variables exist before restoring. """
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
    
    with tf.Session() as sess:
        if model_type in ['downscaler', 'both']:
            # Create downscaler before restoring the model
            downscaler_inputs, downscaler_outputs = create_downscaler(1, model_mod)
            if model_mod == 'IRN':
                load_model(sess, checkpoint.replace("upscaler", "downscaler"), "downscaler")

            image = load_image(image_path, mode='RGB')
            image_input = np.expand_dims(image, axis=0)  # Add batch dimension

            downscaled = sess.run(downscaler_outputs, feed_dict={downscaler_inputs: image_input})
            downscaled = np.squeeze(downscaled)

            if model_type == 'downscaler':
                downscaled_3ch = np.mean(downscaled, axis=-1, keepdims=True).repeat(3, axis=-1)  # Convert to 3-channel
                save_image(downscaled_3ch, "downscaled.png")

        if model_type in ['upscaler', 'both']:
            # Create upscaler before restoring the model
            upscaler_inputs, upscaler_outputs = create_upscaler(1, model_mod)
            load_model(sess, checkpoint, "upscaler")

            if model_type == 'both':
                upscale_input = np.expand_dims(downscaled, axis=0)  # Upscale from downscaled 4-channel
            elif model_type == 'upscaler' and model_mod == 'IRN':
                image = load_image(image_path, mode='RGBA')
                upscale_input = np.expand_dims(image, axis=0)  # Direct upscale
            else:
                image = load_image(image_path, mode='RGB')
                upscale_input = np.expand_dims(image, axis=0)  # Direct upscale               
            upscaled = sess.run(upscaler_outputs, feed_dict={upscaler_inputs: upscale_input})
            upscaled = np.squeeze(upscaled)
            save_image(upscaled, "reconstructed.png" if model_type == 'both' else "upscaled.png")

        if evaluate and model_type == 'both':
            original = load_image(image_path, mode='RGB')
            psnr_value, ssim_value = evaluate_metrics(original, upscaled)
            print(f"PSNR: {psnr_value:.2f}, SSIM: {ssim_value:.4f}")


if __name__ == "__main__":
    inference("infer_config.yml")