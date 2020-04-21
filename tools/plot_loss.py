import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sys
from utils import smooth
import time
import argparse
NUM_IMAGE = 82783
batch_size = 8
num_step_per_epoch = np.ceil(NUM_IMAGE * 1.0 / batch_size)

def get_loss_old(f, which_loss, base_step, ploss_weight, sloss_weight, iloss_weight, plot_style_line):
  step = []
  loss = []
  cnt = -1
  lines = open(f).readlines()
  for line in open(f):
    cnt += 1
    line = line.strip()
    if line.endswith("s/step)"):
      if plot_style_line:
        step_id = line.split(" ")[1]
        if cnt != 0 and (step_id in lines[cnt-1]):
          epoch = int(line.split("E")[1].split("S")[0])
          step_ = epoch * num_step_per_epoch + int(line.split("S")[1].split(" ")[0]) + base_step
          step.append(step_)
          lw = 1 if which_loss == "loss0" else sloss_weight
          loss.append(float(line.split(which_loss)[1].split("=")[1].split(" ")[0].strip()) / lw)
      else:
        step_id = line.split(" ")[1]
        if cnt < len(lines)-1 and (step_id in lines[cnt+1]):
          epoch = int(line.split("E")[1].split("S")[0])
          step_ = epoch * num_step_per_epoch + int(line.split("S")[1].split(" ")[0]) + base_step
          step.append(step_)
          lw = 1 if which_loss == "loss0" else ploss_weight
          loss.append(float(line.split(which_loss)[1].split("=")[1].split(" ")[0].strip()) / lw)
  return np.array(step), np.array(loss)

if __name__ == "__main__":
  
  # Passed-in params
  parser = argparse.ArgumentParser(description="Plot loss")
  parser.add_argument('--which_loss', type=str) # this is the loss mark in the log file, e.g., "loss0" or "iloss"
  parser.add_argument('--log1', type=str)
  parser.add_argument('--log2', type=str)
  parser.add_argument('--base_step', type=str, default="E0S0-E0S0")
  parser.add_argument('--ploss_weight', type=str, default="1.0-1.0")
  parser.add_argument('--sloss_weight', type=str, default="1.0-1.0")
  parser.add_argument('--iloss_weight', type=str, default="1.0-1.0")
  parser.add_argument('--plot_style_line', action='store_true')
  args = parser.parse_args()

  which_loss = args.which_loss.split("-")
  
  base_step1, base_step2 = args.base_step.split("-")
  E1 = int(base_step1.split("E")[1].split("S")[0])
  E2 = int(base_step2.split("E")[1].split("S")[0])
  S1 = int(base_step1.strip().split("S")[1])
  S2 = int(base_step2.strip().split("S")[1])
  base_step1 = E1 * num_step_per_epoch + S1
  base_step2 = E2 * num_step_per_epoch + S2
  base_step = [base_step1, base_step2]

  ploss_weight = [float(i) for i in args.ploss_weight.split("-")]
  sloss_weight = [float(i) for i in args.sloss_weight.split("-")]
  iloss_weight = [float(i) for i in args.iloss_weight.split("-")]

  nloss = len(which_loss)
  cnt = 1
  for wl in which_loss:
    step1, loss1 = get_loss_old(args.log1, wl, base_step[0], ploss_weight[0], sloss_weight[0], iloss_weight[0], args.plot_style_line)
    step2, loss2 = get_loss_old(args.log2, wl, base_step[1], ploss_weight[1], sloss_weight[0], iloss_weight[1], args.plot_style_line)
    plt.subplot(nloss, 1, cnt)
    plt.plot(step1, np.array(smooth(loss1)), label=args.log1.split("Experiments")[1].split("weights")[0]+args.log1.split("weights")[1])
    plt.plot(step2, np.array(smooth(loss2)), label=args.log2.split("Experiments")[1].split("weights")[0]+args.log1.split("weights")[1])
    cnt += 1
    plt.xlabel("step"); plt.ylabel(wl)
    if wl == "loss1":
      plt.ylim([0, 0.03])
    plt.grid(1)
    if wl == which_loss[-1]:
      plt.legend(loc="upper left")
  plt.savefig("%s_%s.png" % (time.strftime("%Y%m%d-%H%M"), args.which_loss))
