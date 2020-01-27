from model import *
from Ex1 import *
import cv2

BUILD_DATASET = False
BATCH_SIZE = 8
NUM_EPOCHS = 1
TRAIN = False
LABELS = {
    0: "ape",
    1: "benchvise",
    2: "cam",
    3: "cat",
    4: "duck"
}
def convert_full_batch(data):
    num_class, num_images, H, W, C = data.shape
    full_batch = np.reshape(data, (-1, H, W, C))
    
    labels = np.array([])
    for i in range(num_class):
        labels = np.hstack((labels, np.full(num_images, i)))

    return full_batch, labels

def main():
    dataset = S_train_images = S_train_poses = S_test_images = S_test_poses = S_db_images = S_db_poses = None
    if BUILD_DATASET:
        print("Building dataset...")
        S_train_images, S_train_poses, S_test_images, S_test_poses, S_db_images, S_db_poses = get_datasets()
        
        num_class, num_images_per_class = np.array(S_train_images).shape[0:2]
        train_set_size = num_class * num_images_per_class
        print("Training images:", train_set_size)

        # TODO: Perhaps without replacement
        batch_count = int(train_set_size / BATCH_SIZE)
        dataset = batch_generator(S_train_images, S_train_poses, S_db_images, S_db_poses, BATCH_SIZE, False)
        for _ in range(batch_count-1):
            batch = batch_generator(S_train_images, S_train_poses, S_db_images, S_db_poses, BATCH_SIZE, False)
            dataset = np.vstack((dataset, batch))
        print("Dataset built.")
        print("Saving dataset...")
        np.savez("data", dataset=dataset, S_train_images=S_train_images, S_train_poses=S_train_poses, S_test_images=S_test_images, 
                 S_test_poses=S_test_poses, S_db_images=S_db_images, S_db_poses=S_db_poses)
        print("Dataset saved.")
    else:
        print("Loading dataset...")
        data = np.load("data.npz")
        dataset, S_train_images, S_train_poses, S_test_images, S_test_poses, S_db_images, S_db_poses = data.values()
        print("Dataset loaded.")

    print("GPU in use:", tf.config.experimental.list_physical_devices('GPU'))
    batch_count, batch_size, H, W, C = dataset.shape
    print("Training Dataset Shape:\n"
          "BC x BS x H x W x C:\n"
          "{} x {} x {} x {} x {}".format(batch_count, batch_size, H, W, C))

    if TRAIN:
        model = Sequential([ 
            layers.Conv2D(input_shape = (64, 64, 3), filters=16, kernel_size=(8,8), activation="relu"),
            layers.MaxPool2D((2, 2)),
            layers.Conv2D(filters=7, kernel_size=(5,5), activation="relu"),
            layers.MaxPool2D((2, 2)),
            layers.Flatten(),
            layers.Dense(256, activation=None),
            layers.Dense(16, activation=None)])
    
        optim = optimizers.Adam(0.0001)
        #model.compile(optimizer = optim, loss = triplet_pair_loss)
    
        for i in range(NUM_EPOCHS):
            print("Number of epochs:", i+1)
            train(model, dataset, optim)
            
        model.save('LeNet.h5')
        
    else:
        test_model = models.load_model('LeNet.h5')

        gt_data, gt_labels = convert_full_batch(S_db_images)
        test_data, test_labels = convert_full_batch(S_test_images)

        print("GT data:", gt_data.shape)
        print("Test data:", test_data.shape)

        gt_preds = test_model.predict(gt_data, batch_size=32)
        test_preds = test_model.predict(test_data, batch_size=32)

        print("G preds:", gt_preds.shape)
        print("Test preds:", test_preds.shape)

        bf = cv2.BFMatcher()
        matches = bf.match(test_preds, gt_preds)              # BFMatcher finds exactly one match for each query from test_preds
        matches = sorted(matches, key = lambda x: x.queryIdx) # queryIdx refers to index of test_preds, trainIdx refers to index of gt_preds
        correct = 0.0
        total = len(matches)
        for match in matches:
            print("\nTest Index (queryIndex): {} - Actual Label for Test: {} ({})\n".format(match.queryIdx, test_labels[match.queryIdx], LABELS[test_labels[match.queryIdx]]),
                  "GT Index (trainIndex): {} - Predicted Label for Test: {} ({})\n".format(match.trainIdx, gt_labels[match.trainIdx], LABELS[gt_labels[match.trainIdx]]),
                  "Distance: {}".format(match.distance), sep="")
            correct += test_labels[match.queryIdx] == gt_labels[match.trainIdx]
        acc = correct / total
        print("\nAccuracy: {} ({}/{})".format(acc, correct, total))
        # TODO: 
        # if test_labels[match.queryIdx] == gt_labels[match.trainIdx]
        # -> abs(S_test_poses[match.queryIdx]-S_db_poses[match.trainIdx]) NOTE: S_test_poses -> (5, 707, 4) S_db_poses -> (5, 267, 4)
        # if < 10 hist[0]++
        # if < 20 hist[1]++
        # if < 40 hist[2]++
        # if < 180 hist[3]++
        
if __name__ == "__main__":
    main()