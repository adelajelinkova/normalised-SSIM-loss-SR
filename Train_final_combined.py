import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
import yaml
import logging
import random
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.ndimage import zoom
from PIL import Image
from tensorflow.python.keras.applications import VGG19
from tensorflow.python.keras.models import Model
from tensorflow.image import resize_images
from tensorflow.keras import regularizers

def residual_block(x, filters=64):
    res = tf.layers.conv2d(x, filters=filters, kernel_size=3, strides=1, padding="same", activation=tf.nn.relu)
    res = tf.layers.conv2d(res, filters=filters, kernel_size=3, strides=1, padding="same", activation=None)
    return x + res

def pixel_shuffle(inputs, upscale_factor=2):
    size = tf.shape(inputs)
    batch_size, height, width, channels = size[0], size[1], size[2], size[3]
    channels //= upscale_factor ** 2
    x = tf.reshape(inputs, [batch_size, height, width, upscale_factor, upscale_factor, channels])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    x = tf.reshape(x, [batch_size, height * upscale_factor, width * upscale_factor, channels])
    return x

def invertible_block(x, filters, kernel_size=3, strides=1, l2_reg=1e-4):
    # Split the input into two parts along the channel dimension
    c = x.get_shape()[-1] // 2
    x1, x2 = tf.split(x, [int(c), int(c)], axis=-1)
    
    # Apply transformations to each split with L2 regularization
    f_x2 = tf.layers.conv2d(
        x2, filters=c, kernel_size=kernel_size, strides=strides,
        padding='same', activation=tf.nn.relu,
        kernel_regularizer=regularizers.l2(l2_reg)
    )
    f_x2 = tf.layers.conv2d(
        f_x2, filters=c, kernel_size=kernel_size, strides=strides,
        padding='same', activation=None,
        kernel_regularizer=regularizers.l2(l2_reg)
    )
    
    # Compute output with affine coupling
    y1 = x1 + f_x2
    y2 = x2
    return tf.concat([y1, y2], axis=-1)


# Model building functions
def create_downscaler(num_downscale_steps, mod, l2_reg=1e-4):
    if mod == 'IRN':    
        with tf.variable_scope('downscaler'):
            inputs = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="downscaler_inputs")
            x = inputs
            
            for i in range(num_downscale_steps):
                filters = 64 if i < num_downscale_steps - 1 else 128
                x = tf.layers.conv2d(
                    x, filters=filters, kernel_size=3, strides=2, padding='same',
                    activation=None, kernel_regularizer=regularizers.l2(l2_reg)
                )
                x = invertible_block(x, filters=filters, l2_reg=l2_reg)
            
            outputs = tf.layers.conv2d(
                x, filters=4, kernel_size=3, padding='same', activation=None,
                kernel_regularizer=regularizers.l2(l2_reg)
            )
            
    elif mod == 'SR':
        with tf.variable_scope('downscaler', reuse=tf.AUTO_REUSE):
            inputs = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="downscaler_inputs")
            x = inputs
            for _ in range(num_downscale_steps):
                x = tf.image.resize(x, size=[tf.shape(x)[1] // 2, tf.shape(x)[2] // 2], method='bicubic')
            outputs = x
    else:
        raise ValueError("Invalid model type specified. Choose either 'IRN' or 'SR'.")
    return inputs, outputs


def create_upscaler(num_upscale_steps, mod, l2_reg=1e-4):
    if mod == 'IRN':
        with tf.variable_scope('upscaler'):
            inputs = tf.placeholder(tf.float32, shape=[None, None, None, 4], name="upscaler_inputs")
            x = inputs
            
            for i in range(num_upscale_steps):
                filters = 64 if i < num_upscale_steps - 1 else 32
                x = tf.layers.conv2d_transpose(x, filters=filters, kernel_size=3, strides=2, padding='same', activation=None)
                x = invertible_block(x, filters=filters)
            
            outputs = tf.layers.conv2d_transpose(x, filters=3, kernel_size=3, padding='same', activation=tf.nn.sigmoid)
            
    elif mod == 'SR':
        with tf.variable_scope('upscaler', reuse=tf.AUTO_REUSE):
            inputs = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="upscaler_inputs")
            x = tf.layers.conv2d(inputs, filters=64, kernel_size=3, strides=1, padding="same", activation=tf.nn.relu, kernel_regularizer=regularizers.l2(l2_reg))
            for _ in range(8):
                x = residual_block(x)
            x = tf.layers.conv2d(x, filters=256, kernel_size=3, strides=1, padding="same", activation=tf.nn.relu, kernel_regularizer=regularizers.l2(l2_reg))
            x = pixel_shuffle(x, upscale_factor=2)
            skip_connection = tf.image.resize(inputs, size=[tf.shape(inputs)[1] * 2, tf.shape(inputs)[2] * 2], method='nearest')
            skip_connection = tf.layers.conv2d(skip_connection, filters=64, kernel_size=1, strides=1, padding="same", activation=None, kernel_regularizer=regularizers.l2(l2_reg))
            x = x + skip_connection
            outputs = tf.layers.conv2d(x, filters=3, kernel_size=3, strides=1, padding="same", activation=None, kernel_regularizer=regularizers.l2(l2_reg))
    else:
        raise ValueError("Invalid model type specified. Choose either 'IRN' or 'SR'.")    
    return inputs, outputs


def model_summary(scope_name, logger):
    logger.info(f"\nModel Summary for {scope_name}:")
    total_params = 0
    for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_name):
        shape = var.get_shape()
        params = np.prod([dim.value for dim in shape])
        total_params += params
        logger.info(f"{var.name} | Shape: {shape} | Params: {params}")
    logger.info(f"Total parameters for {scope_name}: {total_params}\n")

def get_options(yaml_path):
    with open(yaml_path, 'r') as stream:
        return yaml.safe_load(stream)

# Data loading function with normalization
def load_data(train_path, val_path, batch_size, shuffle=True):
    def custom_data_generator(directory):
        file_list = [os.path.join(root, fname) for root, _, files in os.walk(directory) for fname in files if fname.lower().endswith(('png', 'jpg', 'jpeg'))]
        while True:
            if shuffle:
                random.shuffle(file_list)
            images = []
            for file_path in file_list:
                image = Image.open(file_path).convert('RGB')
                w, h = image.size
                new_w = (w // 16) * 16
                new_h = (h // 16) * 16
                if (w, h) != (new_w, new_h):  # Resize only if needed
                    image = image.resize((new_w, new_h), Image.BICUBIC)
                image_np = np.array(image) / 255.0
                images.append(image_np)
                if len(images) == batch_size:
                    yield np.array(images)
                    images = []
            if images:
                yield np.array(images)
                images = []
    return custom_data_generator(train_path), custom_data_generator(val_path)

# Image saving function with normalization adjustments
def save_image_batch(image_batch, epoch, batch_index, stage, checkpoint_dir, logger):
    for i in range(image_batch.shape[0]):
        image = image_batch[i]
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        if image.max() <= 1.0:
            image = np.clip(image * 255, 0, 255)
        min_val, max_val = image.min(), image.max()
        if max_val > min_val:
            image = 255 * (image - min_val) / (max_val - min_val + 1e-7)
        image = image.astype(np.uint8)
        img = Image.fromarray(image)
        path = os.path.join(checkpoint_dir, f"{stage}_epoch_{epoch}_batch_{batch_index}_{i}.png")
        img.save(path)
        logger.info(f"Saved {stage} image at: {path}")

# Model saving function
def save_models(sess, downscaler_saver, upscaler_saver, epoch, checkpoint_dir, save_meta, logger):
    if downscaler_saver:  # Check if downscaler_saver is not None
        downscaler_saver.save(sess, os.path.join(checkpoint_dir, f'downscaler_epoch_{epoch}.ckpt'), write_meta_graph=save_meta)
        logger.info(f"Saved downscaler model at epoch {epoch}")
    if upscaler_saver:
        upscaler_saver.save(sess, os.path.join(checkpoint_dir, f'upscaler_epoch_{epoch}.ckpt'), write_meta_graph=save_meta)
        logger.info(f"Saved upscaler model at epoch {epoch}")

def calculate_psnr(original, reconstructed):
    original = np.squeeze(original)
    reconstructed = np.squeeze(reconstructed)
    if original.shape != reconstructed.shape:
        factors = [original.shape[i] / reconstructed.shape[i] for i in range(len(original.shape))]
        reconstructed_resized = zoom(reconstructed, zoom=factors, order=1)
    else:
        reconstructed_resized = reconstructed
    return psnr(original, reconstructed_resized, data_range=1)

def ssim_loss(y_true, y_pred):
    return 1 - ((1 + tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))) / 2)

# Feature extraction using VGG19 model
def get_vgg_features(input_tensor, feat_l1, feat_l2):
    # Resize and scale input as VGG expects
    input_tensor_resized = resize_images(input_tensor, [224, 224])
    input_tensor_scaled = input_tensor_resized * 255.0

    # Define the selected feature layers
    outputs = [vgg_model.get_layer(feat_l1).output, vgg_model.get_layer(feat_l2).output]
    multi_layer_model = Model(inputs=vgg_model.input, outputs=outputs)
    multi_layer_model.trainable = False

    # Get the features from the input tensor
    feature_outputs = multi_layer_model(input_tensor_scaled)
    return feature_outputs

def perceptual_loss(y_true, y_pred, feat_l1, feat_l2, logger=None):
    # Extract features for each layer from both true and predicted images
    y_true_features = get_vgg_features(y_true, feat_l1, feat_l2)
    y_pred_features = get_vgg_features(y_pred, feat_l1, feat_l2)

    # Calculate the perceptual loss for each layer
    layer_losses = []
    for true_feature, pred_feature in zip(y_true_features, y_pred_features):
        feature_difference = tf.square(true_feature - pred_feature)
        layer_loss = tf.reduce_mean(feature_difference)
        layer_losses.append(layer_loss)

        # Optional logging for each layer's perceptual loss
        if logger:
            logger.info(f"Layer perceptual loss: {tf.reduce_mean(layer_loss):.4f}")

    # Combine the layer losses
    total_perceptual_loss = tf.add_n(layer_losses)
    return total_perceptual_loss

def combined_loss_function(original, reconstructed, low_res, downscaled_from_upscaled, 
                           lambda_bwmse, lambda_perc, lambda_fwmse, lambda_ssim,
                           feat_l1, feat_l2):
    backward_mse_loss = tf.losses.mean_squared_error(original, reconstructed)
    perceptual_loss_val = perceptual_loss(original, reconstructed, feat_l1, feat_l2)
    ssim_loss_val = ssim_loss(original, reconstructed)
    forward_mse_loss = tf.losses.mean_squared_error(low_res, downscaled_from_upscaled) # same as tf.reduce_mean(tf.square(low_res - downscaled_from_upscaled))

    total_error = (lambda_bwmse * backward_mse_loss + lambda_perc * perceptual_loss_val +
                   lambda_fwmse * forward_mse_loss + lambda_ssim * ssim_loss_val)
    return total_error, backward_mse_loss, ssim_loss_val, perceptual_loss_val, forward_mse_loss

# Validation function with logging
def validate(sess, val_gen, downscaler_inputs, downscaler_outputs, upscaler_inputs, upscaler_outputs, 
             validation_steps, logger, combined_loss_op, mse_loss_op, ssim_loss_op, perceptual_loss_op, 
             forward_mse_op, batch_placeholder, reconstructed_placeholder, low_res_placeholder, 
             downscaled_from_upscaled_placeholder, opt):

    # Initialize totals for metrics and losses
    total_psnr, total_ssim = 0, 0
    total_bwmse_loss, total_ssim_loss, total_perceptual_loss, total_forward_mse_loss, total_combined_loss = 0, 0, 0, 0, 0

    for idx in range(validation_steps):
        logger.info(f"Validating batch {idx + 1}/{validation_steps}...")  
        batch_np = next(val_gen).astype(np.float32)
        
        # Downscale and upscale using the model
        low_res = sess.run(downscaler_outputs, feed_dict={downscaler_inputs: batch_np})
        reconstructed = sess.run(upscaler_outputs, feed_dict={upscaler_inputs: low_res})

        # Resize reconstructed if necessary to match batch_np shape
        if batch_np.shape != reconstructed.shape:
            scale_factors = [batch_np.shape[i] / reconstructed.shape[i] for i in range(len(batch_np.shape))]
            reconstructed_resized = zoom(reconstructed, zoom=scale_factors, order=1)
        else:
            reconstructed_resized = reconstructed

        # Calculate PSNR
        psnr_value = calculate_psnr(batch_np, reconstructed_resized)
        total_psnr += psnr_value

        # Calculate loss components
        bwmse_loss_value, ssim_loss_value, perceptual_loss_value, forward_mse_value = sess.run(
            [mse_loss_op, ssim_loss_op, perceptual_loss_op, forward_mse_op],
            feed_dict={
                batch_placeholder: batch_np, 
                reconstructed_placeholder: reconstructed_resized,
                low_res_placeholder: low_res,
                downscaled_from_upscaled_placeholder: sess.run(downscaler_outputs, feed_dict={downscaler_inputs: reconstructed_resized})
            }
        )

        # Reconstructing SSIM
        ssim_value = 1 - 2 * ssim_loss_value
        total_ssim += ssim_value

        # Calculate combined loss explicitly
        combined_loss_value = (
            (opt['loss_weights']['alpha'] * bwmse_loss_value if opt['loss_weights']['alpha'] != 0 else 0) +
            (opt['loss_weights']['beta'] * perceptual_loss_value if opt['loss_weights']['beta'] != 0 else 0) +
            (opt['loss_weights']['gamma'] * forward_mse_value if opt['loss_weights']['gamma'] != 0 else 0) +
            (opt['loss_weights']['lambda_ssim'] * ssim_loss_value if opt['loss_weights']['lambda_ssim'] != 0 else 0)
        )

        # Accumulating losses
        total_combined_loss += combined_loss_value
        total_bwmse_loss += bwmse_loss_value if opt['loss_weights']['alpha'] != 0 else 0
        total_ssim_loss += ssim_loss_value if opt['loss_weights']['lambda_ssim'] != 0 else 0
        total_perceptual_loss += perceptual_loss_value if opt['loss_weights']['beta'] != 0 else 0
        total_forward_mse_loss += forward_mse_value if opt['loss_weights']['gamma'] != 0 else 0

    # Calculate averages
    avg_psnr = total_psnr / validation_steps
    avg_ssim = total_ssim / validation_steps
    avg_combined_loss = total_combined_loss / validation_steps
    avg_bwmse_loss = total_bwmse_loss / validation_steps if opt['loss_weights']['alpha'] != 0 else 0
    avg_ssim_loss = total_ssim_loss / validation_steps if opt['loss_weights']['lambda_ssim'] != 0 else 0
    avg_perceptual_loss = total_perceptual_loss / validation_steps if opt['loss_weights']['beta'] != 0 else 0
    avg_forward_mse_loss = total_forward_mse_loss / validation_steps if opt['loss_weights']['gamma'] != 0 else 0

    # Construct the log message dynamically based on non-zero loss terms
    log_message = f"Validation - PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}, Total_Loss: {avg_combined_loss:.4f}"
    if opt['loss_weights']['alpha'] != 0:
        log_message += f", Backward_MSE: {avg_bwmse_loss:.4f}"
    if opt['loss_weights']['lambda_ssim'] != 0:
        log_message += f", SSIM_Loss: {avg_ssim_loss:.4f}"
    if opt['loss_weights']['beta'] != 0:
        log_message += f", Perceptual_Loss: {avg_perceptual_loss:.4f}"
    if opt['loss_weights']['gamma'] != 0:
        log_message += f", Forward_MSE: {avg_forward_mse_loss:.4f}"

    # Log the message
    logger.info(log_message)

    return avg_psnr, avg_ssim, avg_combined_loss

 
# Training function
def train():
    opt = get_options('train_config_tf.yml')
    checkpoint_dir = opt['train']['checkpoint_dir']
    mod = opt['train']['model_type']
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(checkpoint_dir, "training_log.txt")),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # Data preparation
    train_gen, val_gen = load_data(opt['datasets']['train_path'], opt['datasets']['val_path'], opt['datasets']['batch_size'])
    num_train_images = sum(len(files) for _, _, files in os.walk(opt['datasets']['train_path']))
    total_steps = num_train_images // opt['datasets']['batch_size']
    num_val_images = sum(len(files) for _, _, files in os.walk(opt['datasets']['val_path']))
    validation_steps = num_val_images // opt['datasets']['batch_size']

    # Model components
    downscaler_inputs, downscaler_outputs = create_downscaler(opt['model']['num_downscale_steps'],mod)
    upscaler_inputs, upscaler_outputs = create_upscaler(opt['model']['num_downscale_steps'],mod)
    model_summary('downscaler', logger)
    model_summary('upscaler', logger)

    # DEBUG: Resized output for upscaler - without it, there is tensor shape mismatch sometimes
    
    if upscaler_outputs.shape != downscaler_inputs.shape:
        upscaled_resized = tf.image.resize_images(
            upscaler_outputs,
            [tf.shape(downscaler_inputs)[1], tf.shape(downscaler_inputs)[2]],
            method=tf.image.ResizeMethod.BICUBIC
        )
        logger.info("Reshaped upscaled image.")
    else:
        upscaled_resized = upscaler_outputs
        logger.info("Upscaled image shape perserved.")

    # Combined loss function
    combined_loss, backward_mse_loss, ssim_loss_val, perceptual_loss_val, forward_mse_loss = combined_loss_function(
        downscaler_inputs, upscaled_resized, downscaler_outputs, downscaled_from_upscaled=downscaler_outputs,
        lambda_bwmse=opt['loss_weights']['alpha'], 
        lambda_perc=opt['loss_weights']['beta'], 
        lambda_fwmse=opt['loss_weights']['gamma'], 
        lambda_ssim=opt['loss_weights']['lambda_ssim'],
        feat_l1=opt['vgg_feature_layers']['feature_layer1'], 
        feat_l2=opt['vgg_feature_layers']['feature_layer2']
    )

    # Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=opt['train']['lr'])
    train_op = optimizer.minimize(combined_loss)

    # Validation placeholders and combined loss
    batch_placeholder = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="batch_placeholder")
    reconstructed_placeholder = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="reconstructed_placeholder")
    # downscaled image has shape=[None, None, None, 4] for IRN and [None, None, None, 3] for SR-ResNet
    if mod=='SR':
        low_res_placeholder = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="low_res_placeholder")
        downscaled_from_upscaled_placeholder = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="downscaled_from_upscaled_placeholder")
        logger.info(f"Model selected: {mod}")
    elif mod=='IRN':
        low_res_placeholder = tf.placeholder(tf.float32, shape=[None, None, None, 4], name="low_res_placeholder")
        downscaled_from_upscaled_placeholder = tf.placeholder(tf.float32, shape=[None, None, None, 4], name="downscaled_from_upscaled_placeholder")
        logger.info(f"Model selected: {mod}")
    else:
        logger.info("Model selected incorrectly - has to be either 'SR' or 'IRN'.")
    combined_loss_op, mse_loss_op, ssim_loss_op, perceptual_loss_op, forward_mse_op = combined_loss_function(
        batch_placeholder, reconstructed_placeholder, low_res_placeholder, downscaled_from_upscaled_placeholder,
        lambda_bwmse=opt['loss_weights']['alpha'], 
        lambda_perc=opt['loss_weights']['beta'], 
        lambda_fwmse=opt['loss_weights']['gamma'], 
        lambda_ssim=opt['loss_weights']['lambda_ssim'],
        feat_l1=opt['vgg_feature_layers']['feature_layer1'], 
        feat_l2=opt['vgg_feature_layers']['feature_layer2']
    )

    # Saver objects
    # Get trainable variables for each model
    downscaler_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='downscaler')
    upscaler_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='upscaler')

    # Initialize Saver objects only if variables exist
    downscaler_saver = tf.train.Saver(var_list=downscaler_vars) if downscaler_vars else None
    upscaler_saver = tf.train.Saver(var_list=upscaler_vars)  # Always create upscaler saver
    # Training and validation loop
    with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        sess.run(tf.global_variables_initializer())
        tf.get_default_graph().finalize()

        logger.info("Total steps (batches per epoch): %d", total_steps)

        # Load checkpoints if specified
        if opt['train']['checkpoint_continue']['downscaler']:
            downscaler_saver.restore(sess, opt['train']['checkpoint_continue']['downscaler'])
            logger.info(f"Loaded downscaler from checkpoint: {opt['train']['checkpoint_continue']['downscaler']}")
        if opt['train']['checkpoint_continue']['upscaler']:
            upscaler_saver.restore(sess, opt['train']['checkpoint_continue']['upscaler'])
            logger.info(f"Loaded upscaler from checkpoint: {opt['train']['checkpoint_continue']['upscaler']}")

        for epoch in range(opt['train']['epochs']):
            logger.info(f"Starting epoch {epoch + 1}/{opt['train']['epochs']}")
            for i in range(total_steps):
                batch = next(train_gen)
                low_res = sess.run(downscaler_outputs, feed_dict={downscaler_inputs: batch})
                
                try:
                    _, loss_value, bwmse_loss_value, ssim_loss_value, perceptual_loss_value, forward_mse_value = sess.run(
                        [train_op, combined_loss, backward_mse_loss, ssim_loss_val, perceptual_loss_val, forward_mse_loss],
                        feed_dict={downscaler_inputs: batch, upscaler_inputs: low_res}
                    )
                except tf.errors.ResourceExhaustedError:
                    print("Out of memory during this step, skipping...")
                    continue
                
                # Save images at the start of every epoch
                if i == 0:
                    save_image_batch(batch, epoch, i, "original", checkpoint_dir, logger)
                    save_image_batch(low_res, epoch, i, "downscaled", checkpoint_dir, logger)
                    reconstructed = sess.run(upscaler_outputs, feed_dict={upscaler_inputs: low_res})
                    save_image_batch(reconstructed, epoch, i, "upscaled", checkpoint_dir, logger)

                # Conditional logging based on lambda values
                log_message = (
                    f"Epoch [{epoch + 1}/{opt['train']['epochs']}], Step [{i + 1}/{total_steps}], Total_Loss: {loss_value:.4f}"
                )
                if opt['loss_weights']['alpha'] != 0:
                    log_message += f", Backward_MSE: {bwmse_loss_value:.4f}"
                if opt['loss_weights']['lambda_ssim'] != 0:
                    log_message += f", SSIM_Loss: {ssim_loss_value:.4f}"
                if opt['loss_weights']['beta'] != 0:
                    log_message += f", Perceptual_Loss: {perceptual_loss_value:.4f}"
                if opt['loss_weights']['gamma'] != 0:
                    log_message += f", Forward_MSE: {forward_mse_value:.4f}"

                # PSNR and SSIM calculations
                ssim_value = 1 - 2 * ssim_loss_value
                psnr_value = calculate_psnr(batch, reconstructed)
                log_message += f", PSNR: {psnr_value:.2f}, SSIM: {ssim_value:.4f}"
                
                # Tensor shapes info
                #log_message += f", Input shape: {batch.shape}, Downscaled shape: {low_res.shape}, Reconstructed shape: {reconstructed.shape}"

                # Log the message
                logger.info(log_message)
            
            # Validation after each epoch
            avg_psnr, avg_ssim, avg_loss = validate(
                sess, val_gen, downscaler_inputs, downscaler_outputs,
                upscaler_inputs, upscaler_outputs, validation_steps, logger,
                combined_loss_op, mse_loss_op, ssim_loss_op, perceptual_loss_op, forward_mse_op,
                batch_placeholder, reconstructed_placeholder, low_res_placeholder, downscaled_from_upscaled_placeholder,
                opt  # Pass opt to validate
            )
            logger.info(f"Validation - Epoch [{epoch + 1}/{opt['train']['epochs']}], Loss: {avg_loss:.4f}, PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}")

            # Save checkpoints
            if (epoch + 1) % opt['train']['save_model_freq'] == 0:
                save_meta = epoch == 0
                save_models(sess, downscaler_saver, upscaler_saver, epoch + 1, checkpoint_dir, save_meta, logger)

        logger.info("Training completed!")


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)
    tf.reset_default_graph()
    global vgg_model
    vgg_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    vgg_model.trainable = False
    for layer in vgg_model.layers:
        layer.trainable = False
    train()
