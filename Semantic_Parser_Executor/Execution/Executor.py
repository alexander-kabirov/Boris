from Semantic_Parser_Executor.Parser.Model import Model
from collections import deque
from Semantic_Parser_Executor.Execution.Functions import Functions
import inspect
import pandas as pd


#function_dict['test_print']('test here!')
#model = Model()
#test_input = 'What are unique priority?'
#print(test_input,'is parsed to',model.parse(test_input))
#'name', 'the', 'smallest', 'domestic', 'with', '0', 'cats'

class Executor:
    def __init__(self,dataframes):
        self.dataframes = dataframes #that's a dictionary
        functions = Functions()
        all_functions = inspect.getmembers(functions, predicate=inspect.ismethod)
        self.function_dict = dict((key, value) for key, value in all_functions)
        self.pop_queue = self.function_dict['pop_queue']
        self.model = Model()

    def execute(self,text,df_name):
        rep, ent = self.model.parse(text)
        print(rep,ent)
        rep = rep[1:-1]
        df = self.dataframes[df_name]
        for key in ent:
            if 'col' in key:
                ent[key] = df[ent[key]]
            else:
                try: #convert to number if applicable
                    ent[key] = float(ent[key])
                except ValueError:
                    pass
        ent['df_name'] = df
        ent.update(self.function_dict)
        rep = [ent.get(item, item) for item in rep]
        execution_queue = deque(rep)
        #print('Execution queue:',execution_queue)
        return self.pop_queue(execution_queue)

