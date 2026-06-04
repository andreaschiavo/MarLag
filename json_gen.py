import json

class json_generator:
    def __init__(self, data = None, kwargs = None, args = None):
        self.file = {}
        if data != None:
            if type(data)==dict:
                self.file = data
            else:
                raise ValueError("data must be a dictionary")
        if kwargs is None:
            None
        else:
            if type(kwargs) == str or type(kwargs) == int or type(kwargs) == float:
                self.file.update({kwargs:args})
            elif type(args) == int or type(args) == float or type(args) == str:
                raise ValueError("kwargs is a tuple or list while args is a %s" % type(args))
            elif (len(kwargs) != len(args)):
                raise ValueError("Lenght of kwargs and args doesn't match")
            else:
                for key, value in zip(kwargs, args):
                    self.file.update({key:value})
    
    def add_element(self, kwargs, args):
        if type(kwargs) == str or type(kwargs) == int or type(kwargs) == float:
            self.file.update({kwargs:args})
        elif type(args) == int or type(args) == float or type(args) == str:
            raise ValueError("kwargs is a tuple or list while args is a %s" % type(args))
        elif (len(kwargs) != len(args)):
            raise ValueError("Lenght of kwargs and args doesn't match")
        else:
            for key, value in zip(kwargs, args):
                self.file.update({key:value})
    
    def see(self):
        return self.file
    
    def savefile(self,filename,path='default'):
        filename = str(filename)
        if path !='default':
            filename = path+filename
        
        with open(filename + '.json', 'w') as fp:
            json.dump(self.file, fp)
        
        print("File saved at %s path" % path)

        prova  = json_generator()
        prova.see()