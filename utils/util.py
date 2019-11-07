import os



def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def write_(expr_dir,phase,message):
    if not os.path.exists(expr_dir):
        os.mkdir(expr_dir)
    file_name = os.path.join(expr_dir, '{}.txt'.format(phase))
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')