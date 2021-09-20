def set_opt_param(optimizer, key, value):
    for group in optimizer.param_groups:
        group[key] = value