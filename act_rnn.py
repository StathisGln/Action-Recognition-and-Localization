import torch
import torch.nn as nn


class Act_RNN(nn.Module):
    def __init__(self,  n_inputs, n_neurons, n_outputs):
        super(Act_RNN, self).__init__()
        
        self.n_neurons = n_neurons

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        
        self.basic_rnn = nn.RNN(self.n_inputs, self.n_neurons).cuda()
        
        self.FC = nn.Linear(self.n_neurons, self.n_outputs).cuda()
        
    #     self.hidden = self.init_hidden()
        
    # def init_hidden(self,):
    #     # (num_layers, batch_size, n_neurons)
    #     return (torch.zeros(1, self.batch_size, self.n_neurons))
        
    def forward(self, X, hidden):
        # transforms X to dimensions: n_steps X batch_size X n_inputs

        X = X.unsqueeze(0).permute(1,0,2)
        self.basic_rnn.flatten_parameters()
        nn_out, hidden = self.basic_rnn(X, hidden)
        out = self.FC(hidden)
        
        return out.view(-1, self.n_outputs) # batch_size X n_output

if __name__ == '__main__':

    model = Act_RNN(512, 128, 21).cuda()
    model.hidden = torch.zeros(1,1,128).cuda()
    feats = torch.zeros(3,512).cuda()

    out = model(feats, torch.zeros(1,1,128).cuda())
    print(out.shape)
    print('out :',out)
    
