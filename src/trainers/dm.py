import torch.optim as optim


class Trainer:

    def __init__(self, model, n_epoch, data_loader):
        self.model = model
        self.optim = optim.Adam(self.model.parameters(), lr=1e-4)
        self.data_loader = data_loader
        self.n_epoch = n_epoch

    def train(self):

        self.model.train()
        for epoch in range(1, self.n_epoch + 1):
            for i, (source_data, target_data, source_labels) in enumerate(self.data_loader):
                self.optim.zero_grad()

                f_s, f_t = self.model(source_data, target_data, source_labels)

                loss = self.model.loss
                loss.backward()
                self.optim.step()
            print("EPOCH: {} LOSSS: {}".format(epoch, loss))
