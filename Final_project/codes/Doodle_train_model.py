#This script train the model using transfer learning with the pre-trained Inception-v3 model.  

import glob
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from tqdm import tqdm
import sys

# Training Data parameters
MODEL_DIR = 'model/'  # the path for the inception-v3 model
MODEL_FILE = 'tensorflow_inception_graph.pb'  # inception-v3 model name
CACHE_DIR = 'data/tmp/bottleneck'  # path for feature vectors of the images
INPUT_DATA = 'data/doodle_images'  # path for the training data
VALIDATION_PERCENTAGE = 30  # partion of validation data


# Inception-v3 parameters
BOTTLENECK_TENSOR_SIZE = 2048  # bottleneck node size of inception-v3
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'  # bottleneck tensor name of inception-v3
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'  # images input tensor name

# Neural Network training parameters
LEARNING_RATE = 0.001
STEPS = 100000 #epoches
BATCH = 256 #batch size
CHECKPOINT_EVERY = 100 #save a checkpoint for every n epochs
NUM_CHECKPOINTS = 5 #number of saved checkpoints
EARLY_STOP_PATIENCE = 2000 #early stop patience

# Get all of the images and separate to training/validation data
def create_image_lists(validation_percentage):
    result = {}  # Use a dictionary to save all the images. Keys are class names and values are image names.
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]  # get every sub dir
    is_root_dir = True  # omit the root dir

    
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue

        # get all of images
        extensions = {'jpg', 'jpeg'}
        file_list = []  # store images
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.' + extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list:
            continue

        # separate to training/validation data randomly
        label_name = dir_name.lower() #get the class name
        training_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)  # get the image name
            training_images.append(base_name)
            chance = np.random.randint(100) 
            if chance < validation_percentage:
                validation_images.append(base_name)

        # save the separation
        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'validation': validation_images
        }

    return result


# get the path of an image by its class, dataset class and id.
def get_image_path(image_lists, image_dir, label_name, index, category):
    label_lists = image_lists[label_name]  # get all images of the class
    category_list = label_lists[category]  # get all training/validation images of the class
    mod_index = index % len(category_list)  
    base_name = category_list[mod_index]  # get the id
    sub_dir = label_lists['dir']  
    full_path = os.path.join(image_dir, sub_dir, base_name)  # get the absolute path of the image
    return full_path


# get the path of a bottleneck by its class, dataset class and id.
def get_bottleneck_path(image_lists, label_name, index, category):
    return get_image_path(image_lists, CACHE_DIR, label_name, index,
                          category) + '.txt'


# use inception-v3 to get feature vectors of an image
def run_bottleneck_on_image(sess, image_data, image_data_tensor,
                            bottleneck_tensor):
    bottleneck_values = sess.run(bottleneck_tensor,
                                 {image_data_tensor: image_data})
    bottleneck_values = np.squeeze(bottleneck_values)  # Squeeze 4-D array into 1-D array.
    return bottleneck_values


# get or create a bottleneck
def get_or_create_bottleneck(sess, image_lists, label_name, index, category,
                             jpeg_data_tensor, bottleneck_tensor):
    # get the path of the feature vector of an image
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(CACHE_DIR, sub_dir)
    if not os.path.exists(sub_dir_path):
        os.makedirs(sub_dir_path)
    bottleneck_path = get_bottleneck_path(image_lists, label_name, index,
                                          category)

    # if the feature vector file doesn't exist, then calculate and save it
    if not os.path.exists(bottleneck_path):
        image_path = get_image_path(image_lists, INPUT_DATA, label_name, index,
                                    category)  # get the original image path
        image_data = gfile.FastGFile(image_path, 'rb').read()
        bottleneck_values = run_bottleneck_on_image(
            sess, image_data, jpeg_data_tensor,
            bottleneck_tensor)  # calculate feature vectors with inception-v3

        # save it
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path, 'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)
    else:
        # read the feature vectors from the file
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]

    return bottleneck_values


# get a random batch as the training data
def get_random_cached_bottlenecks(sess, n_classes, image_lists, how_many,
                                  category, jpeg_data_tensor,
                                  bottleneck_tensor):
    bottlenecks = []
    ground_truths = []

    for _ in range(how_many):
        # get a random class and a random id
        label_index = random.randrange(n_classes)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(65535)
        bottleneck = get_or_create_bottleneck(
            sess, image_lists, label_name, image_index, category,
            jpeg_data_tensor, bottleneck_tensor)
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)
    return bottlenecks, ground_truths

#the main function
def main(_):
    # read all of the images
    image_lists = create_image_lists(VALIDATION_PERCENTAGE)
    np.save('imglist.npy',image_lists)
    n_classes = len(image_lists.keys())

    with tf.Graph().as_default() as graph:
        # read the pre-trained inception-v3 model
        with gfile.FastGFile(os.path.join(MODEL_DIR, MODEL_FILE), 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            # load the model and return the input data tensor and the output bottleneck tensor.
            bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(
                graph_def,
                return_elements=[
                    BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME
                ])

        # define the output tensor
        bottleneck_input = tf.placeholder(
            tf.float32, [None, BOTTLENECK_TENSOR_SIZE],
            name='BottleneckInputPlaceholder')

        # define the input tensor
        ground_truth_input = tf.placeholder(
            tf.float32, [None, n_classes], name='GroundTruthInput')

#############################################################################

        ## the last full-connection layer
        with tf.name_scope('final_training_ops'):
            weights = tf.Variable(
                tf.truncated_normal(
                    [BOTTLENECK_TENSOR_SIZE, n_classes], stddev=0.1))
            biases = tf.Variable(tf.zeros([n_classes]))
            logits = tf.matmul(bottleneck_input, weights) + biases
            final_tensor = tf.nn.softmax(logits)
        #learning rate decay
        global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = tf.train.polynomial_decay(
                        learning_rate=LEARNING_RATE, global_step=global_step, decay_steps=22000,
                        end_learning_rate=0.00001, power=0.1, cycle=True)

        # cross-entropy function
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=ground_truth_input)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(
            cross_entropy_mean, global_step)

        # calculate the scores
        with tf.name_scope('evaluation'):
            #accuracy
            correct_prediction = tf.equal(
                tf.argmax(final_tensor, 1), tf.argmax(ground_truth_input, 1))
            evaluation_step = tf.reduce_mean(
                tf.cast(correct_prediction, tf.float32))
            #Mean Average Precision of top 3 classes
            yy = tf.argmax(ground_truth_input, 1)
            topk = tf.nn.top_k(logits, k = 3)
            boo = tf.nn.in_top_k(logits, yy,  k = 3)
            evaluation_step_top3 = tf.reduce_mean(
                tf.cast(boo, tf.float32))

    # training process
    with tf.Session(graph=graph) as sess:
        init = tf.global_variables_initializer().run()

        # save the model and summary
        import time
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(
            os.path.join(os.path.curdir, 'runs', timestamp))
        print('\nWriting to {}\n'.format(out_dir))
        
        # summary of loss function, accuracy and MAP@3
        loss_summary = tf.summary.scalar('loss', cross_entropy_mean)
        acc_summary = tf.summary.scalar('accuracy', evaluation_step)
        acc_summary_top3 = tf.summary.scalar('accuracy', evaluation_step_top3)

        # summary of training 
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, acc_summary_top3])
        train_summary_dir = os.path.join(out_dir, 'summaries', 'train')
        train_summary_writer = tf.summary.FileWriter(train_summary_dir,
                                                     sess.graph)
        # summary of development
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary, acc_summary_top3])
        dev_summary_dir = os.path.join(out_dir, 'summaries', 'dev')
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)
        
        # save the checkpoints
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, 'checkpoints'))
        checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(
                tf.global_variables(), max_to_keep=NUM_CHECKPOINTS)
        
    #early stop parameters
        best_validation_loss = sys.maxsize
        current_epoch = sys.maxsize
        
        # save labels to numpy files
        dict_lables = {}
        keys = list(image_lists.keys())
        for i in range(len(keys)):
            dict_lables[i] = keys[i]
        np.save('keys_labels.npy', dict_lables)

        output_labels = os.path.join(out_dir, 'labels.txt')
        with tf.gfile.FastGFile(output_labels, 'w') as f:
            keys = list(image_lists.keys())
            for i in range(len(keys)):
                keys[i] = '%2d -> %s' % (i, keys[i])
            f.write('\n'.join(keys) + '\n')
        for i in tqdm(range(STEPS)):
            # get a batch of training data every steps
            train_bottlenecks, train_ground_truth = get_random_cached_bottlenecks(
                sess, n_classes, image_lists, BATCH, 'training',
                jpeg_data_tensor, bottleneck_tensor)
            _, train_summaries, train_loss, train_acc,train_acc_top3 = sess.run(
                [train_step, train_summary_op, cross_entropy_mean, evaluation_step, evaluation_step_top3],
                feed_dict={
                    bottleneck_input: train_bottlenecks,
                    ground_truth_input: train_ground_truth
                })
            if i % 10 == 0 or i + 1 == STEPS:
                print('Loss value of training {}\n'.format(train_loss))
                print('Training accuracy {}\n'.format(train_acc))
                print('Training MAP@3 {}\n'.format(train_acc_top3))
            
            # save the summary
            train_summary_writer.add_summary(train_summaries, i)
            steps, lr = sess.run([global_step, learning_rate])
            if i % 10 == 0 or i + 1 == STEPS:
                print('Total steps: ', steps)
                print('Learning rate: ', lr)
            # test the model on the validation data
            if i % 20 == 0 or i + 1 == STEPS:
                validation_bottlenecks, validation_ground_truth = get_random_cached_bottlenecks(
                    sess, n_classes, image_lists, BATCH, 'validation',
                    jpeg_data_tensor, bottleneck_tensor)
                validation_accuracy, validation_accuracy_top3, dev_summaries, validation_loss = \
                  sess.run(
                    [evaluation_step, evaluation_step_top3, dev_summary_op, cross_entropy_mean],
                    feed_dict={
                        bottleneck_input: validation_bottlenecks,
                        ground_truth_input: validation_ground_truth
                    })
                print('Loss value of evaluation is {}\n'.format(validation_loss))
                print('Step %d : Validation accuracy on random sampled %d examples = %.1f%%'% (i, BATCH, validation_accuracy * 100))
                print('Step %d : Validation MAP@3 on random sampled %d examples = %.1f%%'% (i, BATCH, validation_accuracy_top3 * 100))

            
            # save summary of model and validation every n steps
            if i % CHECKPOINT_EVERY == 0:
                dev_summary_writer.add_summary(dev_summaries, i)
                path = saver.save(sess, checkpoint_prefix, global_step=i)
                print('Saved model checkpoint to {}\n'.format(path))
                
            #Early stop
            if (validation_loss < best_validation_loss) and (validation_loss < 3):
                print("Best validation loss: ", validation_loss)
                best_validation_loss = validation_loss
                current_epoch = i
                weight, bias = sess.run([weights, biases])
                np.savez('best_val.npz',k_weight = weight, k_bias = bias)
            elif (i - current_epoch) >= EARLY_STOP_PATIENCE:
                print('early stopping')
                break


if __name__ == '__main__':
    tf.app.run()
