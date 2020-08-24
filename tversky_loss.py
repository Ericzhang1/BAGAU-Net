import torch

#The following ios the orginal tverysky loss implemented in keras/tensorflow
'''
def tversky(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)

def focal_tversky(y_true,y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1-pt_1), gamma)
'''

def forward(input, target, alpha=0.7, smooth=1):    
    input_pos = input.view(-1)
    target_pos = target.view(-1)
    true_pos = torch.sum(input_pos * target_pos)
    false_neg = torch.sum(target_pos * (1 - input_pos))
    false_pos = torch.sum((1 - target_pos) * input_pos)
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)

def tversky_coeff(input, target, alpha):
    """tversky coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + forward(c[0], c[1], alpha=alpha)

    return s / (i + 1)
