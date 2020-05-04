import torch.nn as nn
from model import model_original 
from model import model_cd

class TrainSE_With_WCTDecoder(nn.Module):
  def __init__(self, args):
    super(TrainSE_With_WCTDecoder, self).__init__()
    self.BE = eval("model_original.Encoder%d" % args.stage)(args.BE, fixed=True)
    self.BD = eval("model_original.Decoder%d" % args.stage)(args.BD, fixed=True)
    self.SE = eval("model_cd.SmallEncoder%d_%dx_aux" % (args.stage, args.speedup))(args.SE, fixed=False)
    self.args = args
    
  def forward(self, c, iter):
    cF_BE = self.BE.forward_branch(c) # BE forward, multi outputs: relu1_1, 2_1, 3_1, 4_1, 5_1
    cF_SE = self.SE.forward_aux(c, self.args.updim_relu) # SE forward, multi outputs: [relu1_1, 2_1, 3_1, 4_1, 5_1]
    rec = self.BD(cF_SE[-1])
    
    # for log
    sd_BE = 0
    if iter % self.args.save_interval == 0:
      rec_BE = self.BD(cF_BE[-1])
    
    # (loss 1) BE -> SE knowledge transfer loss
    feat_loss = 0
    for i in range(len(cF_BE)):
      feat_loss += nn.MSELoss()(cF_SE[i], cF_BE[i].data)
    
    # (loss 2, 3) eval the quality of reconstructed image, pixel and perceptual loss
    rec_pixl_loss = nn.MSELoss()(rec, c.data)
    recF_BE = self.BE.forward_branch(rec)
    rec_perc_loss = 0
    for i in range(len(recF_BE)):
      rec_perc_loss += nn.MSELoss()(recF_BE[i], cF_BE[i].data)
    return feat_loss, rec_pixl_loss, rec_perc_loss, rec, c

class TrainSD_With_WCTSE(nn.Module):
  def __init__(self, args):
    super(TrainSD_With_WCTSE, self).__init__()
    self.BE = eval("Encoder%d" % args.stage)(args.BE, fixed=True) 
    self.SE = eval("SmallEncoder%d_%dx_aux" % (args.stage, args.speedup))(args.SE, fixed=True)
    self.SD = eval("SmallDecoder%d_%dx" % (args.stage, args.speedup))(args.SD, fixed=False)
    self.args = args
    
  def forward(self, c, iter):
    rec = self.SD(self.SE(c))
    # loss (1) pixel loss
    rec_pixl_loss = nn.MSELoss()(rec, c.data)
    
    # loss (2) perceptual loss
    recF_BE = self.BE.forward_branch(rec)
    cF_BE = self.BE.forward_branch(c)
    rec_perc_loss = 0
    for i in range(len(recF_BE)):
      rec_perc_loss += nn.MSELoss()(recF_BE[i], cF_BE[i].data)
      
    return rec_pixl_loss, rec_perc_loss, rec

class TrainSD_With_WCTSE_KD2SD(nn.Module):
  def __init__(self, args):
    super(TrainSD_With_WCTSE_KD2SD, self).__init__()
    self.BE = eval("model_original.Encoder%d" % args.stage)(args.BE, fixed=True)
    self.BD = eval("model_original.Decoder%d" % args.stage)(None, fixed=True)
    self.SE = eval("model_cd.SmallEncoder%d_%dx_aux" % (args.stage, args.speedup))(None, fixed=True)
    self.SD = eval("model_cd.SmallDecoder%d_%dx_aux" % (args.stage, args.speedup))(args.SD, fixed=False)
    self.args = args
    
  def forward(self, c, iter):
    feats_BE = self.BE.forward_branch(c) # for perceptual loss

    *_, feat_SE_aux, feat_SE = self.SE.forward_aux2(c) # output the last up-size feature and normal-size feature
    feats_BD = self.BD.forward_branch(feat_SE_aux)
    feats_SD = self.SD.forward_aux(feat_SE, relu=self.args.updim_relu)
    rec = feats_SD[-1]

    # loss (1) pixel loss
    rec_pixl_loss = nn.MSELoss()(rec, c.data)
    
    # loss (2) perceptual loss
    rec_feats_BE = self.BE.forward_branch(rec)
    rec_perc_loss = 0
    for i in range(len(rec_feats_BE)):
      rec_perc_loss += nn.MSELoss()(rec_feats_BE[i], feats_BE[i].data)
    
    # loss (3) kd feature loss
    kd_feat_loss = 0
    for i in range(len(feats_BD)):
      kd_feat_loss += nn.MSELoss()(feats_SD[i], feats_BD[i].data)

    return rec_pixl_loss, rec_perc_loss, kd_feat_loss, rec