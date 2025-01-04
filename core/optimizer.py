from torch.optim import Adam, SGD


def get_optim(opt, parameters, model='adam'):
    if model == 'adam':
        optimizer = Adam(filter(lambda p: p.requires_grad, parameters),
                        lr=opt.learning_rate,
                        weight_decay=opt.weight_decay)
    if model == 'sgd':
        optimizer = SGD(filter(lambda p: p.requires_grad, parameters),
                        lr=opt.learning_rate,
                        weight_decay=opt.weight_decay)
    return optimizer
