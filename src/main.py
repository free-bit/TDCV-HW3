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
    dataset = CustomDataset(BUILD_DATASET)
    print('Length of dataset: ', len(dataset))
    print("GPU in use:", torch.cuda.is_available())
    train_loader = DataLoader(dataset, 
                              batch_size=BATCH_SIZE,
                              shuffle=True, 
                              num_workers=4)
    print(len(train_loader))
    model = TripletNet()
    print(model)
    return
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