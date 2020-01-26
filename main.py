from model import *
from Ex1 import *
import cv2

BATCH_SIZE = 8
NUM_EPOCHS = 1
TRAIN = False

def main():
    S_train_images, S_train_poses, S_test_images, S_test_poses, S_db_images, S_db_poses = get_datasets()
    
    num_class, num_images_per_class = np.array(S_train_images).shape[0:2]
    train_set_size = num_class*num_images_per_class
    print("Training images:", train_set_size)

    # TODO: Perhaps without replacement
    batch_count = int(train_set_size / BATCH_SIZE)
    dataset = batch_generator(S_train_images, S_train_poses, S_db_images, S_db_poses, BATCH_SIZE, False)
    for _ in range(batch_count-1):
        batch = batch_generator(S_train_images, S_train_poses, S_db_images, S_db_poses, BATCH_SIZE, False)
        dataset = np.vstack((dataset, batch))

    print("GPU in use:", tf.config.experimental.list_physical_devices('GPU'))
    print("Batch Count:", batch_count)
    print("Dataset Shape:", dataset.shape)

    
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
        model.compile(optimizer = optim, loss = triplet_pair_loss)
    
        
        for i in range(NUM_EPOCHS):
            print("Number of epochs:", i+1)
            train(model, dataset, optim)
            
        model.save('LeNet.h5')
        
    else:
        test_model = models.load_model('LeNet.h5')
        
        gt_preds = test_model(S_db_images, training=False)
        print('GT_Shape:', gt_preds.shape)
        test_preds = test_model(S_test_images, training = False)
        #bf = cv2.BFMatcher(test_preds[0], gt_preds)
        
        

if __name__ == "__main__":
    main()