import inspect

class Inspector:
    def load_vars(self, *args):
        frame = inspect.currentframe().f_back.f_back   # go one frame up = hello's frame
        vars_caller = frame.f_locals
        return tuple(vars_caller[arg] for arg in args)
    # end

    def load_func(self, arg):
        frame = inspect.currentframe().f_back.f_back
        vars_caller = frame.f_locals
        var_self = vars_caller['self']
        return (var_self, getattr(var_self, arg))
    # end

# end

class HelloWorldInspector(Inspector):
    def __call__(self, *args, **kwds):
        var_self, world = self.load_func('world')
        var = self.load_vars('var')[0]
        world(var)
    # end
# end



class Hello:
    def world(self, var):
        print(var)
    # end

    def hello(self, hook):
        var = 1
        hook()
    # end
# end




if __name__ == '__main__':
    hello = Hello()
    hook = HelloWorldInspector()
    hello.hello(hook)

# end

