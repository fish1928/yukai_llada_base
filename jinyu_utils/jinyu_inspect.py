import inspect

def jinyu_inspect_file(func_target, readfile=False):
    path_file_func = inspect.getsourcefile(func_target)
    print('start to inspect: {}\n\n'.format(path_file_func))

    if readfile:
        with open(path_file_func, 'r') as file:
            print(file.read())
        # end with
    # end
# end
