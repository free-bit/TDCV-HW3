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

def train(model, optim, loss_fn, device, 
          train_loader, db_loader, test_loader, 
          num_epochs=10, log_at=10, draw_hist_at=1000):
    
    total_iter = len(train_loader)

    for epoch in range(1, num_epochs+1):
        print("[Epoch {}/{}]".format(epoch, num_epochs))
        for iter, batch in enumerate(train_loader, 1):
            batch.to(device)

            # TODO: For testing purposes
            # fig = plt.figure()
            # fig.add_subplot(1, 3, 1)
            # plt.imshow(batch[0])

            # fig.add_subplot(1, 3, 2)
            # plt.imshow(batch[1])
            
            # fig.add_subplot(1, 3, 3)
            # plt.imshow(batch[2])
            # plt.show()

            # fig = plt.figure()
            # fig.add_subplot(1, 3, 1)
            # plt.imshow(batch[3])

            # fig.add_subplot(1, 3, 2)
            # plt.imshow(batch[4])
            
            # fig.add_subplot(1, 3, 3)
            # plt.imshow(batch[5])
            # plt.show()
            # END TODO

            optim.zero_grad()                                 # Clear gradients
            preds = model(batch)                              # Get predictions
            loss = loss_fn(preds)                             # Calculate triplet-pair loss
            loss.backward()                                   # Backpropagation
            optim.step()                                      # Optimize parameters based on backpropagation
            # self.train_loss_history.append(loss.item())     # Store loss for each batch

            # Logging in log_at iteration
            if iter % log_at == 0:
                print("[Iteration {}/{}] TRAIN loss: {}".format(iter, total_iter, loss.item()))
            # Drawing histogram in draw_hist_at iteration
            # TODO:

def main():
    train_dataset = CustomDataset(type="train", build=BUILD_DATASET, load=(not BUILD_DATASET))
    train_loader = DataLoader(train_dataset, 
                              batch_size=BATCH_SIZE*3,
                              shuffle=False, 
                              num_workers=4)

    db_dataset = CustomDataset(type="db", build=False, load=False) # Do not build or load
    db_dataset.data_copy(train_dataset)  # Instead, take relevant parts from train_dataset without recomputing
    db_loader = DataLoader(db_dataset, 
                           batch_size=BATCH_SIZE,
                           shuffle=False,
                           num_workers=4)

    test_dataset = CustomDataset(type="test", build=False, load=False) # Do not build or load
    test_dataset.data_copy(train_dataset) # Instead, take relevant parts from train_dataset without recomputing
    test_loader = DataLoader(test_dataset, 
                             batch_size=BATCH_SIZE,
                             shuffle=False, 
                             num_workers=4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device in use:", device)
    print('Length of train dataset: ', len(train_dataset))
    print("Training batch count:", len(train_loader))
    print('Length of DB dataset: ', len(db_dataset))
    print("DB batch count:", len(db_loader))
    print('Length of test dataset: ', len(test_dataset))
    print("Test batch count:", len(test_loader))
    model = TripletNet()
    model.to(device)
    print(model)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = triplet_pair_loss

    train(model, optim, loss_fn, device, train_loader, db_loader, test_loader)
    return # TODO remove later
"""
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
"""
if __name__ == "__main__":
    main()