from model import *
from utils import *

BUILD_DATASET = False
BATCH_SIZE = 32
BATCH_SIZE_TEST = 32
BATCH_SIZE_DB = 32
NUM_EPOCHS = 1
TRAIN = False
LABELS = {
    0: "ape",
    1: "benchvise",
    2: "cam",
    3: "cat",
    4: "duck"
}

# TODO: ask this
def angle_between(v1, v2):
    # Get unit quaternions
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    # Get angle
    return np.arccos(np.abs(np.dot(unit_v1, unit_v2)))

def get_histogram(model, device, test_loader, db_loader):
    model.eval()

    # NOTE: Due to OOM, full-batch was infeasible. Instead, mini-batches are used, with torch.no_grad() blocks added and results of batched are concatenated.

    # Forward pass with test images to get descriptor vectors
    test_preds = torch.empty(0).type(torch.FloatTensor)
    test_labels = torch.empty(0).type(torch.FloatTensor)
    test_poses = torch.empty(0).type(torch.DoubleTensor)
    with torch.no_grad():
        for images, labels, poses in test_loader:
            images = images.to(device)
            new_preds = model(images)
            test_preds = torch.cat((test_preds, new_preds.cpu()))
            test_labels = torch.cat((test_labels, labels))
            test_poses = torch.cat((test_poses, poses))

    with torch.no_grad():
        # Forward pass with db images to get descriptor vectors
        db_preds = torch.empty(0).type(torch.FloatTensor)
        db_labels = torch.empty(0).type(torch.FloatTensor)
        db_poses = torch.empty(0).type(torch.DoubleTensor)
        for images, labels, poses in db_loader:
            images = images.to(device)
            new_preds = model(images)
            db_preds = torch.cat((db_preds, new_preds.cpu()))
            db_labels = torch.cat((db_labels, labels))
            db_poses = torch.cat((db_poses, poses))

    # Convert all PyTorch tensors at GPU to numpy arrays at CPU
    test_preds = test_preds.cpu().numpy()
    test_labels = test_labels.cpu().numpy().astype(int)
    test_poses = test_poses.cpu().numpy()
    db_preds = db_preds.cpu().numpy()
    db_labels = db_labels.cpu().numpy().astype(int)
    db_poses = db_poses.cpu().numpy()
    
    # Brute-force matching
    bf = cv2.BFMatcher()
    matches = bf.match(test_preds, db_preds)              # BFMatcher finds exactly one match for each query from test_preds
    matches = sorted(matches, key = lambda x: x.queryIdx) # queryIdx refers to index of test_preds, trainIdx refers to index of db_preds

    # TODO: Accuracy calculation will be removed (#)
    hist = np.zeros(4)
    correct = 0.0  #
    total = len(matches)  #
    for match in matches:
        print("\nTest Index (queryIndex): {} - Actual Label for Test: {} ({})\n".format(match.queryIdx, test_labels[match.queryIdx], LABELS[test_labels[match.queryIdx]]),
                "GT Index (trainIndex): {} - Predicted Label for Test: {} ({})\n".format(match.trainIdx, db_labels[match.trainIdx], LABELS[db_labels[match.trainIdx]]),
                "Distance: {}".format(match.distance), sep="")  #
        if test_labels[match.queryIdx] == db_labels[match.trainIdx]:
            angle_diff = angle_between(test_poses[match.queryIdx], db_poses[match.trainIdx])
            # TODO: Fix error in angle differences
            if angle_diff < 10:
                hist[0] += 1
            if angle_diff < 20:
                hist[1] += 1
            if angle_diff < 40:
                hist[2] += 1
            if angle_diff < 180:
                hist[3] += 1

            correct += 1 #
    acc = correct / total  #
    print("\nAccuracy: {} ({}/{})".format(acc, correct, total))  #
    print("Histogram:", hist)
    model.train()
    return hist

def train(model, optim, loss_fn, device,
          train_loader, test_loader, db_loader, 
          num_epochs=10, log_at=10, draw_hist_at=1000):
    
    total_iter = len(train_loader)
    total_global_iter = total_iter * num_epochs
    global_iter_count = 1

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
                print("[Total iterations {}/{}] loss: {}\n".format(global_iter_count, total_global_iter, loss.item()),
                      "Calculating histogram...", sep="")
                get_histogram(model, device, test_loader, db_loader)
            global_iter_count += 1

def main():
    train_dataset = CustomDataset(type="train", build=BUILD_DATASET)
    train_loader = DataLoader(train_dataset, 
                              batch_size=BATCH_SIZE*3,
                              shuffle=False, 
                              num_workers=4)

    db_dataset = CustomDataset(type="db", build=False, copy_from=train_dataset) # Do not build or load, take data from train_dataset without recomputing
    db_loader = DataLoader(db_dataset, 
                           batch_size=BATCH_SIZE_TEST,
                           shuffle=False,
                           num_workers=4)

    test_dataset = CustomDataset(type="test", build=False, copy_from=train_dataset) # Do not build or load, take data from train_dataset without recomputing
    test_loader = DataLoader(test_dataset, 
                             batch_size=BATCH_SIZE_DB,
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

    train(model, optim, loss_fn, device, train_loader, test_loader, db_loader, num_epochs=5)

    if not os.path.exists("../models"):
        os.makedirs('../models')
    torch.save(model, "../models/triplet.model")
    #model = torch.load('../models/triplet.model')

if __name__ == "__main__":
    main()