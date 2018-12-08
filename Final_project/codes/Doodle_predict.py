import tensorflow as tf
import numpy as np
import heapq
from glob import glob
import pandas as pd
from tqdm import tqdm

# model dir
CHECKPOINT_DIR = './runs/1543117935/checkpoints'
INCEPTION_MODEL_FILE = 'model/tensorflow_inception_graph.pb'

# inception-v3 model parameters
BOTTLENECK_TENSOR_SIZE = 2048  # bottleneck node size of inception-v3
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'  # bottleneck tensor name of inception-v3
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'  # images input tensor name

# test data path
file_path = glob('./TestImages/*.jpg')

n_classes = 340

key_ids = []#store key ids
top3_predictions = []#store top3 predictions
# traversal all of test images
for i in range(len(file_path)):
    image_data = tf.gfile.FastGFile(file_path[i], 'rb').read()

    key_id = file_path[i][13:-4]
    key_ids.append(key_id)

# Make the prediciton
    checkpoint_file = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    with tf.Graph().as_default() as graph:
        # read the pre-trained inception-v3 model
        with tf.Session().as_default() as sess:
            with tf.gfile.FastGFile(INCEPTION_MODEL_FILE, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                # load the model and return the input data tensor and the output bottleneck tensor.
                bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(
                                                                      graph_def,
                                                                      return_elements=[
                                                                                       BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME
                                                                                       ])
                                                                          
            # define the output tensor
                bottleneck_values = sess.run(bottleneck_tensor, {jpeg_data_tensor: image_data})
                bottleneck_values = [np.squeeze(bottleneck_values)]
                bottleneck_input = tf.placeholder(
                                              tf.float32, [None, BOTTLENECK_TENSOR_SIZE],
                                              name='BottleneckInputPlaceholder')
            
            # define the input tensor
                ground_truth_input = tf.placeholder(tf.float32, [None, n_classes], name='GroundTruthInput')
            
            

            #the last full-connection layer
                with tf.name_scope('final_training_ops'):
                    weights = tf.constant(data['k_weight'])
                    biases = tf.constant(data['k_bias'])
                    logits = tf.matmul(bottleneck_input, weights) + biases
                    final_tensor = tf.nn.softmax(logits)

                    
            # collect the predictions
                data = np.load('best_val.npz')
                label = np.load('keys_labels.npy')#load the label list
                all_predictions = []
                all_predictions = sess.run(final_tensor,
                               {bottleneck_input: bottleneck_values})

                predictions = all_predictions[0].tolist()
                label = label.tolist()
                max_num_index_list = list(map(predictions.index, heapq.nlargest(3, predictions)))

                #put three classes in one string
                res = ''
                for idx in max_num_index_list:
                    res = res + label[idx] + ' '
                res = res[:-1]
                
    top3_predictions.append(res)

#save the prediction in csv file
key_ids = pd.DataFrame(key_ids,columns=['key_id'])
top3_predictions = pd.DataFrame(top3_predictions,columns=['word'])
result = pd.concat([key_ids,top3_predictions],axis=1)
result.to_csv('submission.csv',index=False)
