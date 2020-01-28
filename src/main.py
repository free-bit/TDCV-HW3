from model import *
from utils import *

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

def get_histogram(model, ...):
    model.eval()
    gt_preds = model()
    test_preds = model()

    bf = cv2.BFMatcher()
    matches = bf.match(test_preds, gt_preds)              # BFMatcher finds exactly one match for each query from test_preds
    matches = sorted(matches, key = lambda x: x.queryIdx) # queryIdx refers to index of test_preds, trainIdx refers to index of gt_preds

    # TODO: Accuracy calculation will be removed
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
    # hist = np.zeros(4)
    # if test_labels[match.queryIdx] == gt_labels[match.trainIdx]
    # -> abs(S_test_poses[match.queryIdx]-S_db_poses[match.trainIdx]) NOTE: S_test_poses -> (5, 707, 4) S_db_poses -> (5, 267, 4)
    # if < 10 hist[0]++
    # if < 20 hist[1]++
    # if < 40 hist[2]++
    # if < 180 hist[3]++
    model.train()

def train(model, optim, loss_fn, device, 
          train_loader, db_loader, test_loader, 
          num_epochs=10, log_at=10, draw_hist_at=1000):
    
    total_iter = len(train_loader)
    total_global_iter = total_iter * num_epochs
    global_iter_count = 0

    for epoch in range(1, num_epochs+1):
        print("[Epoch {}/{}]".format(epoch, num_epochs))
        for iter, batch in enumerate(train_loader, 1):
            batch = batch.to(device)

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

            optim.zero_grad()                             # Clear gradients
            preds = model(batch)                          # Get predictions
            loss = loss_fn(preds)                         # Calculate triplet-pair loss
            loss.backward()                               # Backpropagation
            optim.step()                                  # Optimize parameters based on backpropagation
            # self.train_loss_history.append(loss.item()) # Store loss for each batch

            # Logging in log_at iteration
            if iter % log_at == 0:
                print("[Iteration {}/{}] loss: {}".format(iter, total_iter, loss.item()))
            # Drawing histogram in draw_hist_at iteration
            if global_iter_count % draw_hist_at == 0:
                print("[Total iterations {}/{}] loss: {}".format(global_iter_count, total_global_iter, loss.item()))
                # TODO:
            global_iter_count += 1

def main():
    train_dataset = CustomDataset(type="train", build=BUILD_DATASET)
    train_loader = DataLoader(train_dataset, 
                              batch_size=BATCH_SIZE*3,
                              shuffle=False, 
                              num_workers=4)

    db_dataset = CustomDataset(type="db", build=False, copy_from=train_dataset) # Do not build or load, take data from train_dataset without recomputing
    db_loader = DataLoader(db_dataset, 
                           batch_size=BATCH_SIZE,
                           shuffle=False,
                           num_workers=4)

    test_dataset = CustomDataset(type="test", build=False, copy_from=train_dataset) # Do not build or load, take data from train_dataset without recomputing
    test_loader = DataLoader(test_dataset, 
                             batch_size=BATCH_SIZE,
                             shuffle=False, 
                             num_workers=4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device in use:", device)

    train_dataset.print_datasets()

    model = TripletNet()
    model.to(device)
    print(model)
    
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = triplet_pair_loss

    train(model, optim, loss_fn, device, train_loader, db_loader, test_loader, num_epochs=2)

    if not os.path.exists("../models"):
        os.makedirs('../models')
    torch.save(model, "../models/triplet.model")
    #model = torch.load('../models/triplet.model')


    return # TODO remove later

if __name__ == "__main__":
    main()