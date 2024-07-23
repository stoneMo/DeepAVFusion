import math


def adjust_learning_rate(optimizer, epoch, args):
    wu = args.opt.get('warmup_epochs', 0)
    if epoch < wu:
        lr = args.opt.lr * epoch / wu
    else:
        lr = args.opt.lr * 0.5 * (1. + math.cos(math.pi * (epoch - wu) / (args.opt.epochs - wu)))

    # Learning rate adjustment for pretrained components
    pt_warmup_epochs = eval(str(args.opt.get('pt_warmup_epochs', -1)))
    if epoch < pt_warmup_epochs:
        lr_pt_scale = (0.5 - 0.5 * math.cos(math.pi * epoch / pt_warmup_epochs)) * (args.opt.pt_lr_mult_end - args.opt.pt_lr_mult_start) + args.opt.pt_lr_mult_start
    else:
        lr_pt_scale = args.opt.get('pt_lr_mult_end', 1.)

    for param_group in optimizer.param_groups:
        lr_layer_scale = param_group.get('lr_scale', 1.)
        if param_group.get('pretrained', False):
            param_group["lr"] = lr * lr_layer_scale * lr_pt_scale
        else:
            param_group["lr"] = lr * lr_layer_scale
    return lr


def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_groups = {}

    weights_layer_id = {k: v for k, v in model.params_layer_ids()}
    num_layers = max(list(weights_layer_id.values()))
    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay

        group_name = "layer_%d_%s" % (weights_layer_id[p], g_decay)
        if group_name not in param_groups:
            param_groups[group_name] = {
                "lr_scale": layer_scales[weights_layer_id[p]],
                "weight_decay": this_decay,
                "params": [],
            }
        param_groups[group_name]["params"].append(p)

    return list(param_groups.values())


def get_layer_id_for_vit(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if name in ['cls_token', 'pos_embed']:
        return 0
    elif name.startswith('patch_embed'):
        return 0
    elif name.startswith('blocks'):
        return int(name.split('.')[1]) + 1
    else:
        return num_layers


def param_groups_pretrained(model, weight_decay=0.05, no_weight_decay_list=[], image_pt=None, audio_pt=None):
    from timm.optim import optim_factory
    param_groups = optim_factory.param_groups_weight_decay(
        model, weight_decay,
        no_weight_decay_list=no_weight_decay_list)
    param_groups_pt = []
    if image_pt is not None:
        param_groups_pt += optim_factory.param_groups_weight_decay(model.encoder.image, weight_decay, no_weight_decay_list=no_weight_decay_list)
    if audio_pt is not None:
        param_groups_pt += optim_factory.param_groups_weight_decay(model.encoder.audio, weight_decay, no_weight_decay_list=no_weight_decay_list)
    for group in param_groups_pt:
        group['pretrained'] = True
    params_pt = set([p for group in param_groups_pt for p in group['params']])
    for group in param_groups:
        group['params'] = [p for p in group['params'] if p not in params_pt]
    param_groups += param_groups_pt
    return param_groups