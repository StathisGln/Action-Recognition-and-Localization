import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    filename = 'images_etc/training_rnn.txt'
    # filename = './nohup.out'

    epoch = []
    loss = []
    epoch_recall = []
    recall_05 = []
    recall_04 = []
    recall_03 = []
    with open(filename, 'r') as fp:
        lines = fp.readlines()
        for i in range(len(lines)):
            l = lines[i]
            if l.startswith('Train Epoch'):
                words = l.split()

                epoch.append(int(words[2]))
                loss.append(float(words[4]))
            if l.startswith('| Validation Epoch'):
                epoch_recall.append( int(l.split()[3]))
                recall_05.append( float(lines[i+12].split()[3]))
                recall_04.append( float(lines[i+18].split()[3]))
                recall_03.append( float(lines[i+24].split()[3]))

                
            
    # for e,l in zip(epoch,loss):
    #     print('epoch :',e, ' loss :',l)
    print('epoch_recall :',epoch_recall)
    print('epoch :',epoch)
    fig = plt.figure()
    ax = plt.subplot(111, label='1')
    ax.plot(epoch[2:],loss[2:], 'r')
    ax.plot(epoch_recall,recall_05, 'g')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('Loss.png')

    fig = plt.figure()

    ax2 = plt.subplot(111, label='1')
    ax2.plot(epoch_recall,recall_05, 'g')

    plt.savefig('Validation.png')

    
