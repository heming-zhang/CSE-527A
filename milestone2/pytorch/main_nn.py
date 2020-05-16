from nn import NNLoad,RunNN
from nn import NNParameter

def run_mnist_nn(nn_batchsize, nn_worker,
            nn_learning_rate, nn_momentum, 
            nn_cuda, nn_epoch_num):
    # LOAD DATA WITH MINI-BATCH
    train_loader = NNLoad(nn_batchsize, nn_worker).load_data()
    # # USE NEURAL NETWORK AS TRAINING MODEL
    model, criterion, optimizer = NNParameter(
                nn_learning_rate,
                nn_momentum,
                nn_cuda).nn_function()
    # GET EXPERIMENTAL RESULTS IN EVERY EPOCH
    train_accuracy_rate = []
    for epoch in range(1, nn_epoch_num + 1):
        train_loss, train_accuracy = RunNN(model,
                    criterion,
                    optimizer,
                    train_loader,
                    nn_cuda).train_nn()
        train_accuracy_rate.append(train_accuracy)
        print("Epoch:{:d}, Train Accuracy:{:7.6f}".format(epoch, train_accuracy))


if __name__ == "__main__":
    # NEURAL NETWORK PARAMETERS
    nn_batchsize = 640
    nn_worker = 2
    nn_learning_rate = 0.01
    nn_momentum = 0.5
    nn_cuda = False
    nn_epoch_num = 50
    run_mnist_nn(nn_batchsize, nn_worker, nn_learning_rate,
            nn_momentum, nn_cuda, nn_epoch_num)
