from model import *
from utils import *
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd


BUILD_DATASET = False
BATCH_SIZE = 32
BATCH_SIZE_TEST = 32
BATCH_SIZE_DB = 32
NUM_EPOCHS = 10
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
    return (2*np.arccos(np.abs(np.dot(unit_v1, unit_v2)))) *180/np.pi

def get_histogram(model, device, test_loader, db_loader, iteration):
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
    total = len(matches)
    pred_labels = []
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
        
        pred_labels.append(db_labels[match.trainIdx])
    pred_labels = np.array(pred_labels)
    acc = correct / total  #
    print("\nAccuracy: {} ({}/{})".format(acc, correct, total))  #
    print("Histogram:", hist)
    
    angle_labels = ['<10°', '<20', '<40', '<180']

    sum_hist = np.sum(hist)
    hist = hist/sum_hist*100
    
    x_pos = [i for i, _ in enumerate(angle_labels)]
    plt.style.use('ggplot')
    plt.bar(x_pos, hist, color='blue')
    plt.xlabel("Angles, $^\circ$")
    plt.ylabel("Percentage, %")
    plt.title("Angle histogram")
    plt.xticks(x_pos, ('<10$^\circ$', '<20$^\circ$', '<40$^\circ$', '<180$^\circ$'))
    plt.yticks(np.arange(0, np.max(hist)+1, 5.))
    plt.savefig('hist_'+ str(iteration) +'.png')
    plt.show()
    
    model.train()
    return hist, pred_labels, test_labels

def train(model, optim, loss_fn, device,
          train_loader, test_loader, db_loader, 
          num_epochs=10, log_at=10, draw_hist_at=1000):
    
    total_iter = len(train_loader)
    total_global_iter = total_iter * num_epochs
    global_iter_count = 1
    with open("loss_log.txt", "w") as file:
        for epoch in range(1, num_epochs+1):
            info = "[Epoch {}/{}]".format(epoch, num_epochs)
            print(info)
            file.write(info + "\n")
            for iter, batch in enumerate(train_loader, 1):
                batch = batch.to(device)
                optim.zero_grad()                             # Clear gradients
                preds = model(batch)                          # Get predictions
                loss = loss_fn(preds)                         # Calculate triplet-pair loss
                loss.backward()                               # Backpropagation
                optim.step()                                  # Optimize parameters based on backpropagation

                # Logging in log_at iteration
                if iter % log_at == 0:
                    info = "[Iteration {}/{}] loss: {}".format(iter, total_iter, loss.item())
                    print(info)
                    file.write(info + "\n")
                # Drawing histogram in draw_hist_at iteration
                if global_iter_count % draw_hist_at == 0:
                    print("[Total iterations {}/{}] loss: {}\n".format(global_iter_count, total_global_iter, loss.item()),
                        "Calculating histogram...", sep="")
                    get_histogram(model, device, test_loader, db_loader, global_iter_count)
                global_iter_count += 1

def main():
    train_dataset = CustomDataset(type="train", build=BUILD_DATASET)
    train_loader = DataLoader(train_dataset, 
                              batch_size=BATCH_SIZE*3,
                              shuffle=False, 
                              num_workers=0)

    db_dataset = CustomDataset(type="db", build=False, copy_from=train_dataset) # Do not build or load, take data from train_dataset without recomputing
    db_loader = DataLoader(db_dataset, 
                           batch_size=BATCH_SIZE_TEST,
                           shuffle=False,
                           num_workers=0)

    test_dataset = CustomDataset(type="test", build=False, copy_from=train_dataset) # Do not build or load, take data from train_dataset without recomputing
    test_loader = DataLoader(test_dataset, 
                             batch_size=BATCH_SIZE_DB,
                             shuffle=False, 
                             num_workers=0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device in use:", device)

    train_dataset.print_datasets()

    model = TripletNet()
    model.to(device)
    print(model)
    
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = triplet_pair_loss

    if TRAIN:
        train(model, optim, loss_fn, device, train_loader, test_loader, db_loader, num_epochs=NUM_EPOCHS)
    
        if not os.path.exists("../models"):
            os.makedirs('../models')
        torch.save(model, "../models/triplet.model")
        
    else:
        model = torch.load('../models/triplet.model')
        hist, pred_labels, true_class = get_histogram(model, device, test_loader, db_loader, 0)
        
        #Check for same size of the lists
        if(len(pred_labels) != len(true_class)):
            print("Unequal shape for Confusion Matrix! Predictions: {}, Truth: {}".format(len(pred_labels), len(true_class)))
        else: # print the confusion matrix
            cm = confusion_matrix(true_class, pred_labels)
            cm = (cm / cm.astype(np.float).sum(axis=1)) * 100 # normalize to get % of the confusion matrix
            df_cm = pd.DataFrame(cm, index = [i for i in LABELS.values()], columns = [i for i in LABELS.values()])
            print(df_cm)
            plt.figure(figsize = (10, 7))
            sn.heatmap(df_cm, annot=True, cmap="Blues")
            # print("Saving the image in follwing path: {}".format(base_folder))
            #plt.savefig('confusion_m_' + str(TOTAL_ITER) + '.png', bbox_inches='tight')
 
if __name__ == "__main__":
    main()