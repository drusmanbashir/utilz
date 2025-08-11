import ast
       

class PrintableDict(dict): # for a dict
    def __str__(self):
        return '. '.join(' '.join(str(x) for x in
            chain.from_iterable(zip((item[0], 'is', 'and'), item[1])))
                for item in self.iteritems()) + '.'
class DictToAttr(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

    def assimilate_dict(self,dic:dict):
        for key,val in dic.items():
            setattr(self,key,val)
 
    @classmethod
    def from_nested_dicts(cls, data):
        """ Construct nested AttrDicts from nested dictionaries. """
        if not isinstance(data, dict):
            return data
        else:
            return cls({key: cls.from_nested_dicts(data[key]) for key in data})


def key_from_value(dici:dict, value):
    keys = []
    for k,v in dici.items():
        if v == value:
            keys.append(k)
    return keys

def assimilate_dict(cls,dic:dict):
        objectified_dic = DictToAttr.from_nested_dicts(dic)
        for key,val in objectified_dic.items():
            setattr(cls,key,val)
        return cls

def list_unique(values:list):
     return list(set(values))


def dic_lists_to_sets(dic:dict)->dict:
        switcher= lambda key,val: [key,set(val)] if isinstance(val,list) else [key,val]
        dic_out={}
        for key,val in dic.items():
            outs = switcher(key,val)
            dic_out.update({outs[0]:outs[1]})
        return dic_out

def dic_in_list(query_dic,dic_list)->bool:
    query_dic = dic_lists_to_sets(query_dic)
    dic_list= [dic_lists_to_sets(dic) for dic in dic_list]
    return True if query_dic in dic_list else False


def fix_ast(dici,keys):
    for key in keys:
        if isinstance(dici[key], str):
            dici[key]= ast.literal_eval(dici[key])
    return dici


