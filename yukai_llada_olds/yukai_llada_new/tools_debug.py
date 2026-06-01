import inspect

def jprint(*args):
    frame = inspect.currentframe().f_back  # f_back = caller's frame
    name_file = frame.f_code.co_filename
    name_func = frame.f_code.co_name
    num_line = frame.f_lineno

    str_target = ' '.join([str(arg) for arg in args])

    print(f'[DEBUG][{name_func}:{num_line}] {str_target} [FILE] {name_file}')
# end
