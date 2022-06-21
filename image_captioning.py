from module import Preprocess, Train

class ImageCaptioning:
    def __init__(self):
        pass


    def print_console(self, message):
        print('-' * 10 + message + '-' * 10)


    def preprocess(self):
        preprocess = Preprocess(DATASET='ms_coco', print_console=self.print_console)
        train_ids, train_captions, img_name_vector, corpus = preprocess.load_dataset()
        #preprocess.feature_extraction(img_name_vector=img_name_vector)
        tokenizer, cap_vector = preprocess.tokenize(train_captions=train_captions)
        _, _, img_name_train, _, cap_train, _ = preprocess.train_test_split(train_ids=train_ids, img_name_vector=img_name_vector, cap_vector=cap_vector)
        return tokenizer, img_name_train, cap_train

    def training(self, tokenizer, num_steps, img_name_train, cap_train):
        trainer = Train(print_console=self.print_console, NUM_STEPS=num_steps)
        trainer.train(tokenizer=tokenizer, img_name_train=img_name_train, cap_train=cap_train)




if __name__ == '__main__':
    captioning = ImageCaptioning()
    tokenizer, img_name_train, cap_train = captioning.preprocess()
    captioning.training(tokenizer=tokenizer, num_steps=len(img_name_train), img_name_train=img_name_train,
                        cap_train=cap_train)