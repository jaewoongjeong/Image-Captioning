import os
import re
import numpy as np
import time
import json
import pickle
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import cv2
from yolov4.tf import YOLOv4
import yolov5
import tensorflow as tf

from model import CNN_Encoder, RNN_Decoder

class Preprocess:
    def __init__(self, print_console, DATASET='ms_coco', DATASET_SIZE = 30000, RESET=True, WORD_DICT_SIZE=5000,
                 TEST_SET_PROPORTION=0.2):
        self.DATASET = DATASET
        self.DATASET_SIZE = DATASET_SIZE
        self.RESET = RESET
        self.WORD_DICT_SIZE = WORD_DICT_SIZE
        self.TEST_SET_PROPORTION = TEST_SET_PROPORTION
        self.print_console = print_console


    # Loads, Shuffles and Extracts the portion of Dataset
    def load_dataset(self):
        # Flickr30k Dataset
        if self.DATASET == 'flickr30k':
            self.print_console('Loading Flickr30k Dataset')

            flickr30k = './Dataset/Flickr30k'

        # MS-COCO Dataset
        else:
            self.print_console('Loading MS-COCO Dataset')

            ms_coco = './Dataset/MS_COCO'
            with open(ms_coco + '/annotations/captions_train2014.json', 'r') as file:
                annotation = json.load(file)

            image_id_index = {}

            all_captions = [] # Reference Sentences of an image
            all_img_name_vector = [] # Path of an image file
            all_ids = []  # Image IDs
            corpus = [] # Reference Sentences without <start> & <end> (for tf-idf)

            for img in annotation['images']:
                image_id_index[img['id']] = img['file_name']

            for annot in annotation['annotations']:
                corpus.append(annot['caption'].lower())
                caption = '<start> ' + annot['caption'] + ' <end>'
                image_id = annot['image_id']
                full_coco_image_path = ms_coco + '/images/train2014/' + image_id_index[image_id]

                all_ids.append(image_id)  # hadie
                all_img_name_vector.append(full_coco_image_path)
                all_captions.append(caption)

            train_ids, train_captions, img_name_vector, corpus = shuffle(all_ids,  # hadie
                                                                         all_captions,
                                                                         all_img_name_vector,
                                                                         corpus,
                                                                         random_state=1)

        self.print_console('Finished Loading Dataset')
        return train_ids[:self.DATASET_SIZE], train_captions[:self.DATASET_SIZE], img_name_vector[:self.DATASET_SIZE],\
               corpus[:self.DATASET_SIZE]


    # Returns the array of Bounding Boxes of identified objects using YOLO
    def bounding_boxes(self, yolov4, directory):
        frame = cv2.imread(directory)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bboxes = yolov4.predict(frame, prob_thresh=0.25)
        bboxes = bboxes.tolist()
        n = len(bboxes)

        # For each bounding box, append (area * confidence)
        for i in range(n):
            bboxes[i].append(bboxes[i][2] * bboxes[i][3] * bboxes[i][5])
        bboxes.sort()
        return np.array(bboxes)


    '''
    def yolo_v5(self):
        # load pretrained model
        yolo = yolov5.load('yolov5s.pt')

        # set model parameters
        yolo.conf = 0.25  # NMS confidence threshold
        yolo.iou = 0.45  # NMS IoU threshold
        yolo.agnostic = False  # NMS class-agnostic
        yolo.multi_label = False  # NMS multiple labels per box
        yolo.max_det = 1000  # maximum number of detections per image

        # set image
        img = 'example.jpg'

        # perform inference
        results = yolo(img)

        # parse results
        predictions = results.pred[0]
        boxes = predictions[:, :4]  # x1, y1, x2, y2
        scores = predictions[:, 4]
        categories = predictions[:, 5]

        # show detection bounding boxes on image
        results.show()
    '''


    # Extraction of Image Features using Inception & YOLO
    def feature_extraction(self, img_name_vector):
        self.print_console('Extracting Features')

        def load_image(image_path):
            img = tf.io.read_file(image_path)
            img = tf.io.decode_jpeg(img, channels=3)
            img = tf.keras.layers.Resizing(299, 299)(img)
            img = tf.keras.applications.inception_v3.preprocess_input(img)
            return img, image_path

        # YOLO v4
        yolov4 = YOLOv4()

        yolov4.config.parse_names('coco.names')
        yolov4.config.parse_cfg('yolov4.cfg')

        yolov4.make_model()
        yolov4.load_weights("yolov4.weights", weights_type="yolo")

        # Inception v3
        image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
        new_input = image_model.input
        hidden_layer = image_model.layers[-1].output

        image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

        # Get unique images
        encode_train = sorted(set(img_name_vector))

        features_shape = 2048

        # Concatenating Inception v3 with YOLO v4
        image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
        image_dataset = image_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(16)

        for img, path in tqdm(image_dataset):
            batch_features = image_features_extract_model(img)
            batch_features = tf.reshape(batch_features,
                                        (batch_features.shape[0], -1, batch_features.shape[3]))

            for bf, p in zip(batch_features, path):
                path_of_feature = p.numpy().decode("utf-8")
                yolo_features = self.bounding_boxes(yolov4, path_of_feature)
                yolo_features = np.array(yolo_features.flatten())
                yolo_features = np.pad(yolo_features, (0, features_shape - yolo_features.shape[0]), 'constant',
                                       constant_values=(0, 0)).astype(np.float32)
                combined_features = np.vstack((bf.numpy(), yolo_features)).astype(np.float32)
                np.save(path_of_feature, combined_features)

        self.print_console('Finished Extracting Features')


    # Tokenizing Dataset
    def tokenize(self, train_captions):
        if self.RESET:
            self.print_console('Tokenizing Captions')

            tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.WORD_DICT_SIZE,
                                                              oov_token="<unk>",
                                                              filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
            tokenizer.fit_on_texts(train_captions)
            tokenizer.word_index['<pad>'] = 0
            tokenizer.index_word[0] = '<pad>'

            with open("tokenizer.pickle", 'wb') as file:
                pickle.dump(tokenizer, file, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            self.print_console('Using Cached Tokenizer')

            with open('tokenizer.pickle', 'rb') as file:
                tokenizer = pickle.load(file)

        # Create the tokenized vectors
        train_seqs = tokenizer.texts_to_sequences(train_captions)

        # Pad each vector to the max_length of the captions
        # If you do not provide a max_length value, pad_sequences calculates it automatically
        cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')

        if self.RESET:
            # Calculates the max_length, which is used to store the attention weights
            max_length = max(len(t) for t in train_seqs)

            file = "max_length.txt"
            with open(file, 'w') as file:
                file.write(str(max_length)) # write the maximum length to disk

        else:
            with open('max_length.txt', 'r') as file:
                max_length = int(file.readline())

        self.print_console('Finished Tokenizing Captions')

        return tokenizer, cap_vector, max_length


    # Splitting into train and validation sets
    def train_test_split(self, train_ids, img_name_vector, cap_vector):
        self.print_console('Splitting Dataset into Train and Validation Sets')
        # image_id: Image ID
        # img_name: Full directory of an Image File
        # cap: Tokenized Caption
        image_id_train, image_id_val, img_name_train, img_name_val, cap_train, cap_val = train_test_split(
            train_ids,
            img_name_vector,
            cap_vector,
            test_size=self.TEST_SET_PROPORTION,
            random_state=0)

        self.print_console('Splitting Dataset into Train and Validation Sets')
        return image_id_train, image_id_val, img_name_train, img_name_val, cap_train, cap_val




class Train:
    def __init__(self, print_console, NUM_STEPS, RESET=True, BATCH_SIZE=64, BUFFER_SIZE=10000, EMBEDDING_DIM=256,
                 UNITS=512, VOCAB_SIZE=5000, EPOCHS=20):
        self.RESET = RESET
        self.BATCH_SIZE = BATCH_SIZE
        self.BUFFER_SIZE = BUFFER_SIZE
        self.EMBEDDING_DIM = EMBEDDING_DIM
        self.UNITS = UNITS
        self.VOCAB_SIZE = VOCAB_SIZE + 1
        self.NUM_STEPS = NUM_STEPS // BATCH_SIZE
        self.EPOCHS = EPOCHS
        self.print_console = print_console


    # Load the numpy files
    def map_func(self, img_name, cap):
        img_tensor = np.load(img_name.decode('utf-8') + '.npy')
        return img_tensor, cap


    # Creates a Dataset for Training
    def make_dataset(self, img_name_train, cap_train):
        self.print_console('Making Dataset for Training')

        dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

        # Use map to load the numpy files in parallel
        dataset = dataset.map(lambda item1, item2: tf.numpy_function(
            self.map_func, [item1, item2], [tf.float32, tf.int64]),
                              num_parallel_calls=tf.data.AUTOTUNE)

        # Shuffle and batch
        dataset = dataset.shuffle(self.BUFFER_SIZE).batch(self.BATCH_SIZE)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        self.print_console('Finished Making Dataset for Training')
        return dataset


    # Loss Function used during Training
    def loss_function(self, real, pred, loss_object):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)


    # Tensorflow Function used for Training
    @tf.function
    def train_step(self, img_tensor, encoder, decoder, optimizer, tokenizer, target):
        loss = 0

        # initializing the hidden state for each batch
        # because the captions are not related from image to image
        hidden = decoder.reset_state(batch_size=target.shape[0])

        dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)

        with tf.GradientTape() as tape:
            features = encoder(img_tensor)

            for i in range(1, target.shape[1]):
                # passing the features through the decoder
                predictions, hidden, _ = decoder(dec_input, features, hidden)

                loss += self.loss_function(target[:, i], predictions)

                # using teacher forcing
                dec_input = tf.expand_dims(target[:, i], 1)

        total_loss = (loss / int(target.shape[1]))

        trainable_variables = encoder.trainable_variables + decoder.trainable_variables

        gradients = tape.gradient(loss, trainable_variables)

        optimizer.apply_gradients(zip(gradients, trainable_variables))

        return loss, total_loss


    # Trains and Stores the Model Weights
    def train(self, tokenizer, dataset):
        encoder = CNN_Encoder(self.EMBEDDING_DIM)
        decoder = RNN_Decoder(self.EMBEDDING_DIM, self.UNITS, self.VOCAB_SIZE)
        optimizer = tf.keras.optimizers.Adam()  # hadie
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')  # hadie

        checkpoint_path = "./checkpoints/train"
        # Deletes all the checkpoints if RESET
        if self.RESET:
            try:
                for filename in os.listdir(checkpoint_path):
                    self.print_console("Deleting " + checkpoint_path + "/" + filename)
                    os.unlink(checkpoint_path + "/" + filename)
            except Exception as e:
                self.print_console(f'Failed to delete {checkpoint_path + "/" + filename}. Reason: {e}')

        ckpt = tf.train.Checkpoint(encoder=encoder, decoder=decoder, optimizer=optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)


        start_epoch = 0
        if ckpt_manager.latest_checkpoint:
            start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
            # restoring the latest checkpoint in checkpoint_path
            ckpt.restore(ckpt_manager.latest_checkpoint)


        loss_plot = []
        if self.RESET:
            self.print_console('Start Training')

            for epoch in range(start_epoch, self.EPOCHS):
                start = time.time()
                total_loss = 0

                for (batch, (img_tensor, target)) in enumerate(dataset):
                    batch_loss, t_loss = self.train_step(img_tensor=img_tensor, encoder=encoder, decoder=decoder,
                                                         optimizer=optimizer, tokenizer=tokenizer, target=target)
                    total_loss += t_loss

                    if batch % 100 == 0:
                        print('Epoch {} Batch {} Loss {:.4f}'.format(
                            epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))
                # storing the epoch end loss value to plot later
                loss_plot.append(total_loss / self.NUM_STEPS)

                if epoch % 5 == 0:
                    ckpt_manager.save()

                print('Epoch {} Loss {:.6f}'.format(epoch + 1,
                                                    total_loss / self.NUM_STEPS))
                print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

            # Saves Model Weights to Disk
            decoder.save_weights("model_weights", save_format="tf")

        else:
            self.print_console('A trained model has been found. Loading it from disk..')

            # Load the previously saved weights
            decoder.load_weights("model_weights")

    ''''''
    # Captioning Image
    def evaluate(self, image):
        attention_plot = np.zeros((max_length, attention_features_shape))

        hidden = decoder.reset_state(batch_size=1)

        temp_input = tf.expand_dims(load_image(image)[0], 0)
        img_tensor_val = image_features_extract_model(temp_input)
        img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0],
                                                     -1,
                                                     img_tensor_val.shape[3]))

        features = encoder(img_tensor_val)

        dec_input = tf.expand_dims([word_to_index(tf.convert_to_tensor('<start>'))], 0)
        result = []

        for i in range(max_length):
            predictions, hidden, attention_weights = decoder(dec_input,
                                                             features,
                                                             hidden)

            attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()
            predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()

            predicted_word = tf.compat.as_text(index_to_word(tf.convert_to_tensor(predicted_id)).numpy())
            result.append(predicted_word)

            if predicted_word == '<end>':
                return result, attention_plot

            dec_input = tf.expand_dims([predicted_id], 0)

        attention_plot = attention_plot[:len(result), :]
        return result, attention_plot

