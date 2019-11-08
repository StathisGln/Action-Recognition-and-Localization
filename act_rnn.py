import torch
import torch.nn as nn


class Act_RNN(nn.Module):
    def __init__(self,  n_inputs, n_neurons, n_outputs):
        super(Act_RNN, self).__init__()
        
        self.n_neurons = n_neurons

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.basic_rnn = nn.RNN(self.n_inputs, self.n_neurons)
        self.FC = nn.Linear(self.n_neurons, self.n_outputs)

    #     self.hidden = self.init_hidden()
        
    # def init_hidden(self,):
    #     # (num_layers, batch_size, n_neurons)
    #     return (torch.zeros(1, self.batch_size, self.n_neurons))
        
    def forward(self, X):
        # transforms X to dimensions: n_steps X batch_size X n_inputs
        self.hidden = torch.zeros(1,1,self.n_neurons).type_as(X)
        X = X.unsqueeze(0).permute(1,0,2)
        self.basic_rnn.flatten_parameters()
        nn_out, self.hidden = self.basic_rnn(X, self.hidden)      
        out = self.FC(self.hidden)
        
        return out.view(-1, self.n_outputs) # batch_size X n_output

if __name__ == '__main__':

    model = Act_RNN(512, 128, 21).cuda()
    model.hidden = torch.zeros(1,1,128).cuda()
    feats = torch.zeros(1,3,512).cuda()

    out = model(feats)
    print(out.shape)

    
