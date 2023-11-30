import numpy as np
import matplotlib.pyplot as plt 
import argparse

start_plot_idx = 1
def parse_args():
   parser = argparse.ArgumentParser(description='show loss acc')
   parser.add_argument('--file', dest='lossfile', help='the model', default='loss.refine', type=str)
   args = parser.parse_args()
   return args

def show_loss(lossfile,statistic_interval,lineset,scale=1.0):
   loss_file = open(lossfile, 'r')
   loss_total = loss_file.readlines()
   loss_num = len(loss_total)
   loss_res = np.zeros(loss_num)
   loss_idx = np.arange(loss_num)

   for idx in range(loss_num) :
       loss_str = loss_total[idx]
       str_start_idx = loss_str.find('tensor(')+6
       str_end_idx = loss_str.find('device')-2
       tmp = loss_str[str_start_idx+1:str_end_idx]
       print "tmp=",tmp
       loss_res[idx] = scale*float(tmp)
       
   statistic_len = (loss_num + statistic_interval - 1)/statistic_interval
   statistic_idx = np.arange(statistic_len) * statistic_interval
   statistic_res_mean = np.zeros(statistic_len)
   statistic_res_var = np.zeros(statistic_len)
   
   for idx in range(statistic_len) :
       loss_start_idx = idx*statistic_interval
       loss_end_idx = min(loss_start_idx + statistic_interval, loss_num)
       loss_part = loss_res[loss_start_idx : loss_end_idx]
       statistic_res_mean[idx] = np.mean(loss_part)
       statistic_res_var[idx] = np.var(loss_part)
       
   plt.plot(statistic_idx[start_plot_idx:], statistic_res_mean[start_plot_idx:],lineset)
if __name__ == '__main__':
    args = parse_args()
    show_loss('ploss.refine',1,'r-')
    show_loss('nloss.refine',1,'g-',scale=10000)
    show_loss('mseloss.refine',1,'b-')
    #show_loss('loss.refine',1,'k-')
    #show_loss('plossz.refine',1,'r--')
    #show_loss('nlossz.refine',1,'g--')
    #show_loss('mselossz.refine',1,'b--')
    #show_loss('lossz.refine',1,'k--')
    #plt.legend(('w perceptual loss','w noise_loss','w mse_loss','w total loss','z perceptual loss','z noise_loss','z mse_loss','z total loss'))
    plt.legend(('w perceptual loss','w noise_loss','w mse_loss','w total loss'))
    plt.title('train_loss')
    plt.xlabel('niters')
    plt.show()
