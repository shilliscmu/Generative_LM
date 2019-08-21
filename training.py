import gc
import math
import numpy as np
from matplotlib import pyplot as plt
import time
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import tests

dataset = np.load('../dataset/wiki.train.npy')
fixtures_pred = np.load('../fixtures/prediction.npz')  # dev
fixtures_gen = np.load('../fixtures/generation.npy')  # dev
fixtures_pred_test = np.load('../fixtures/prediction_test.npz')  # test
fixtures_gen_test = np.load('../fixtures/generation_test.npy')  # test
vocab = np.load('../dataset/vocab.npy')

# data loader

class LanguageModelDataLoader(DataLoader):
    """
        TODO: Define data loader logic here
    """
    def __init__(self, dataset, batch_size, shuffle=True):
        super(DataLoader, self).__init__()
        self.original_data = dataset
        self.shuffle = shuffle
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        # concatenate your articles and build into batches
        if self.shuffle:
            np.random.shuffle(self.dataset)
        words_remaining = True
        current_index = 0

        flat_articles = np.concatenate(self.dataset)
        words_to_drop = len(flat_articles) % self.batch_size
        flat_articles = flat_articles[:-words_to_drop]
        batched_articles = np.reshape(flat_articles, (self.batch_size, -1))

        while words_remaining:
            distribution_chooser = np.random.random_sample()
            if distribution_chooser < 0.05:
                seq_length = math.ceil(np.random.normal(35, 5))
            else:
                seq_length = math.ceil(np.random.normal(70, 5))
            if current_index + seq_length >= batched_articles.shape[1]:
                seq_length = batched_articles.shape[1] - current_index
                words_remaining = False
            data = batched_articles[:, current_index:int(current_index+seq_length)]
            input = torch.LongTensor(data[:, :-1])
            target = torch.LongTensor(data[:, 1:])
            current_index += seq_length
            yield(input, target)


# model

class LanguageModel(nn.Module):
    """
        TODO: Define your model here
    """
    def __init__(self, vocab_size):
        super(LanguageModel, self).__init__()
        self.modules = []
        self.embedding = nn.Embedding(vocab_size, 200)
        # self.modules.append(self.embedding)
        # self.lstm_a = nn.LSTM(input_size=400,
        #                       hidden_size=1150,
        #                       num_layers=1,
        #                       bidirectional=False,
        #                       batch_first=True)
        # self.modules.append(self.lstm_a)
        # self.lstm_b = nn.LSTM(input_size=1150,
        #                       hidden_size=1150,
        #                       num_layers=1,
        #                       bidirectional=False,
        #                       batch_first=True)
        # self.modules.append(self.lstm_b)
        # self.lstm_c = nn.LSTM(input_size=1150,
        #                       hidden_size=400,
        #                       num_layers=1,
        #                       bidirectional=False,
        #                       batch_first=True)
        # self.modules.append(self.lstm_c)
        # self.linear = nn.Linear(400, vocab_size)
        # self.modules.append(self.linear)
        #
        # self.embedding.weight = self.linear.weight

        self.lstm = nn.LSTM(input_size=200, hidden_size=512, num_layers=3, batch_first=True, bidirectional=False)
        self.modules.append(self.lstm)

        self.linear = nn.Linear(512, vocab_size)

        self.net = nn.Sequential(*self.modules)

    def forward(self, x, h_0s=None):
        # Feel free to add extra arguments to forward (like an argument to pass in the hiddens)

        embedded_x = self.embedding(x)
        if h_0s:
            # lstm_a_output, h0_0 = self.lstm_a(embedded_x, h_0s[0])
            # lstm_b_output, h0_1 = self.lstm_b(lstm_a_output, h_0s[1])
            # lstm_c_output, h0_2 = self.lstm_c(lstm_b_output, h_0s[2])
            lstm_output, h0_0 = self.lstm(embedded_x, h_0s[0])
        else:
            # lstm_a_output, h0_0 = self.lstm_a(embedded_x)
            # lstm_b_output, h0_1 = self.lstm_b(lstm_a_output)
            # lstm_c_output, h0_2 = self.lstm_c(lstm_b_output)
            lstm_output, h0_0 = self.lstm(embedded_x)
        output = self.linear(lstm_output)
        # return output, [h0_0, h0_1, h0_2]
        return output, [h0_0]

# model trainer

class LanguageModelTrainer:
    def __init__(self, model, loader, max_epochs=1, run_id='exp'):
        """
            Use this class to train your model
        """
        # feel free to add any other parameters here
        self.model = model
        self.loader = loader
        self.train_losses = []
        self.val_losses = []
        self.predictions = []
        self.predictions_test = []
        self.generated_logits = []
        self.generated = []
        self.generated_logits_test = []
        self.generated_test = []
        self.epochs = 0
        self.max_epochs = max_epochs
        self.run_id = run_id
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), 1e-3, weight_decay=1e-6)
        self.criterion = nn.CrossEntropyLoss(reduction='sum')

    def train(self):
        self.model.train() # set to training mode
        start_time = time.time()
        epoch_loss = 0
        for batch_num, (inputs, targets) in enumerate(self.loader):
            epoch_loss += self.train_batch(inputs, targets)
        epoch_loss = epoch_loss / (batch_num + 1)
        self.epochs += 1
        print('[TRAIN]  Epoch [%d/%d]   Loss: %.4f    Time: %.2f'
                      % (self.epochs + 1, self.max_epochs, epoch_loss, time.time()-start_time))
        self.train_losses.append(epoch_loss)

    def train_batch(self, inputs, targets):
        inputs = inputs.cuda()
        targets = targets.cuda()
        outputs, _ = self.model(inputs)
        loss = self.criterion(outputs.view(-1, outputs.size(2)), targets.view(-1)) / (inputs.shape[0] * inputs.shape[1])
        loss.backward()
        # nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
        self.optimizer.step()
        self.optimizer.zero_grad()
        # gc.collect()
        # torch.cuda.empty_cache()

        return loss
    
    def test(self):
        # don't change these
        self.model.eval() # set to eval mode
        predictions = TestLanguageModel.prediction(fixtures_pred['inp'], self.model) # get predictions
        self.predictions.append(predictions)
        generated_logits = TestLanguageModel.generation(fixtures_gen, 10, self.model) # generated predictions for 10 words
        generated_logits_test = TestLanguageModel.generation(fixtures_gen_test, 10, self.model)
        nll = tests.test_prediction(predictions, fixtures_pred['out'])

        generated = tests.test_generation(fixtures_gen, generated_logits, vocab)
        generated_test = tests.test_generation(fixtures_gen_test, generated_logits_test, vocab)
        self.val_losses.append(nll)
        
        self.generated.append(generated)
        self.generated_test.append(generated_test)
        self.generated_logits.append(generated_logits)
        self.generated_logits_test.append(generated_logits_test)
        
        # generate predictions for test data
        predictions_test = TestLanguageModel.prediction(fixtures_pred_test['inp'], self.model) # get predictions
        self.predictions_test.append(predictions_test)
            
        print('[VAL]  Epoch [%d/%d]   Loss: %.4f'
                      % (self.epochs + 1, self.max_epochs, nll))
        return nll

    def save(self):
        # don't change these
        model_path = os.path.join('experiments', self.run_id, 'model-{}.pkl'.format(self.epochs))
        torch.save({'state_dict': self.model.state_dict()},
            model_path)
        np.save(os.path.join('experiments', self.run_id, 'predictions-{}.npy'.format(self.epochs)), self.predictions[-1])
        np.save(os.path.join('experiments', self.run_id, 'predictions-test-{}.npy'.format(self.epochs)), self.predictions_test[-1])
        np.save(os.path.join('experiments', self.run_id, 'generated_logits-{}.npy'.format(self.epochs)), self.generated_logits[-1])
        np.save(os.path.join('experiments', self.run_id, 'generated_logits-test-{}.npy'.format(self.epochs)), self.generated_logits_test[-1])
        with open(os.path.join('experiments', self.run_id, 'generated-{}.txt'.format(self.epochs)), 'w') as fw:
            fw.write(self.generated[-1])
        with open(os.path.join('experiments', self.run_id, 'generated-{}-test.txt'.format(self.epochs)), 'w') as fw:
            fw.write(self.generated_test[-1])


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Embedding:
        torch.nn.init.xavier_normal_(m.weight)
    elif type(m) == nn.LSTM:
        for name, param in m.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                upper = 1/np.sqrt(m.hidden_size)
                nn.init.uniform_(param, -upper, upper)


class TestLanguageModel:
    def prediction(inp, model):
        """
            :param inp:
            :return: a np.ndarray of logits
        """
        inp = torch.cuda.LongTensor(inp)
        # inp = torch.LongTensor(inp)
        scores, h_0 = model(inp)
        predicted_words = scores[:,-1]
        return predicted_words.cpu().detach().numpy()

        
    def generation(inp, forward, model):
        """
            Generate a sequence of words given a starting sequence.
            :param inp: Initial sequence of words (batch size, length)
            :param forward: number of additional words to generate
            :return: generated words (batch size, forward)
        """
        model.eval()
        with torch.set_grad_enabled(False):
            generated_words = []
            inp = torch.cuda.LongTensor(inp)
            # inp = torch.LongTensor(inp)
            scores, h_0s = model(inp)
            current_word = torch.argmax(scores, dim=2)[:,-1].unsqueeze(1)
            generated_words.append(current_word)
            if forward > 1:
                for i in range(forward-1):
                    scores, h_0s = model(current_word, h_0s)
                    current_word = torch.argmax(scores, dim=2)[:,-1].unsqueeze(1)
                    generated_words.append(current_word)
        return torch.cat(generated_words,dim=1).cpu().detach()


NUM_EPOCHS = 10
BATCH_SIZE = 40

run_id = str(int(time.time()))
if not os.path.exists('./experiments'):
    os.mkdir('./experiments')
os.mkdir('./experiments/%s' % run_id)
print("Saving models, predictions, and generated words to ./experiments/%s" % run_id)

model = LanguageModel(len(vocab))
model.apply(init_weights)
model = model.cuda()
loader = LanguageModelDataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
trainer = LanguageModelTrainer(model=model, loader=loader, max_epochs=NUM_EPOCHS, run_id=run_id)

best_nll = 1e30 
for epoch in range(NUM_EPOCHS):
    # trainer.train()
    nll = trainer.test()
    # gc.collect()
    # torch.cuda.empty_cache()
    if nll < best_nll:
        best_nll = nll
        print("Saving model, predictions and generated output for epoch "+str(epoch)+" with NLL: "+ str(best_nll))
        trainer.save()

# Don't change these
# plot training curves
plt.figure()
plt.plot(range(1, trainer.epochs + 1), trainer.train_losses, label='Training losses')
plt.plot(range(1, trainer.epochs + 1), trainer.val_losses, label='Validation losses')
plt.xlabel('Epochs')
plt.ylabel('NLL')
plt.legend()
plt.show()

# see generated output
print (trainer.generated[-1]) # get last generated output

def log_softmax(x, axis):
    ret = x - np.max(x, axis=axis, keepdims=True)
    lsm = np.log(np.sum(np.exp(ret), axis=axis, keepdims=True))
    return ret - lsm


def array_to_str(arr, vocab):
    return " ".join(vocab[a] for a in arr)


def test_prediction(out, targ):
    out = log_softmax(out, 1)
    nlls = out[np.arange(out.shape[0]), targ]
    nll = -np.mean(nlls)
    return nll

def test_generation(inp, pred, vocab):
    outputs = u""
    for i in range(inp.shape[0]):
        w1 = array_to_str(inp[i], vocab)
        w2 = array_to_str(pred[i], vocab)
        outputs += u"Input | Output #{}: {} | {}\n".format(i, w1, w2)
    return outputs


