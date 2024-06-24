def get_data(path="mfeat-pix.txt"):
    with open(path, "r") as f:
        lines = f.readlines()
    return lines
    

if __name__ == "__main__":
    pass
    # train = []
    # test = []
    # data = get_data()
    # for class_number in range(0, 10):
    #     train_test = data[class_number*200: class_number*200+200]
    #     train.append(train_test[0: 100])
    #     test.append(train_test[100: 200])
    # with open("Training_data.txt", "w") as f:
    #     for class_num in train:
    #         for val in class_num:
    #             f.write(val)
    # with open("Testing_data.txt", "w") as f:
    #     for class_num in test:
    #         for val in class_num:
    #             f.write(val)