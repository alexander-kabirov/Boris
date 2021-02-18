import pandas as pd

class Functions:
    def __init__(self):
        pass

    def pop_queue(self,queue):
        item = queue.popleft()  # pop the queue
        if callable(item):  # callable() checks if it is a function
            item = item(queue)  # pass the remianing of the queue as the variable
        return item

    def f_select(self,queue):
        column_names = self.pop_queue(queue)
        if isinstance(column_names, pd.Series):
            column_names = [column_names]
        df = self.pop_queue(queue)
        return df[[column if isinstance(column, str) else column.name for column in column_names]]

    def f_filter(self,queue):
        df = self.pop_queue(queue)
        criteria = self.pop_queue(queue)
        return df[criteria]

    def f_eq(self,queue):  # >>> Here is an important issue - if filter consumes an already filtered df, the criteria won't work as it operates on the full dataset,
        # so there should be one filter for a query only combining several criteria with f_and and f_or
        # can pop two parameters or can yield a function which takes additional parameter
        return self.pop_queue(queue) == self.pop_queue(queue)

    def f_tuple(self,queue):
        return [self.pop_queue(queue), self.pop_queue(queue)]

    def f_length(self,queue):
        return len(self.pop_queue(queue))

    # and funtion accepts two arguments, for more items it should be nested
    def f_and(self,queue):
        return self.pop_queue(queue) & self.pop_queue(queue)

    def f_or(self,queue):
        return self.pop_queue(queue) | self.pop_queue(queue)

    def f_max(self,queue):
        column_name = self.pop_queue(queue)  # we are taking just the name here because the indexes may change in case of filtering
        column_name = column_name if isinstance(column_name, str) else column_name.name
        df = self.pop_queue(queue)
        column = df[column_name]  # so here we take the column by name from the dataset which could be filtered
        return df.loc[column.idxmax()]

    def f_min(self,queue):
        column_name = self.pop_queue(queue)  # we are taking just the name because the indexes may change in case of filtering
        column_name = column_name if isinstance(column_name, str) else column_name.name
        df = self.pop_queue(queue)
        column = df[column_name]  # so here we take the column by name from the dataset which could be filtered
        return df.loc[column.idxmin()]

    def f_count(self,queue):  # returns dataframe with grouping columns and the new 'count' column, it accepts invidual grouping columns or the ones obtained from the f_tuple function
        column_names = self.pop_queue(queue)
        if isinstance(column_names, pd.Series):
            column_names = [column_names]
        df = self.pop_queue(queue)
        return pd.DataFrame(df.groupby([column.name for column in column_names]).size(),
                            columns=['count']).reset_index()

    def f_ratio(self,queue):
        column_name = self.pop_queue(queue)
        column_name = column_name if isinstance(column_name, str) else column_name.name
        df1 = self.pop_queue(queue)
        df2 = self.pop_queue(queue)
        keys = set(df1.columns).intersection(set(df2.columns))
        keys.discard(column_name)  # get rid of the column used for ratio among the keys
        lsuffix = '_left'
        rsuffix = '_right'
        joined_df = df1.set_index(list(keys)).join(df2.set_index(list(keys)), lsuffix=lsuffix, rsuffix=rsuffix)
        name_l = column_name + lsuffix
        name_r = column_name + rsuffix
        return pd.DataFrame(joined_df[name_l] / joined_df[name_r], columns=['ratio']).reset_index()

    def test_print(self,text):
        print(text)