import json
from torch.utils.data import Dataset  
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sql.sql import SQLStatement, DataBase, SQL_KEYWORDS, SQL_COND_OPS, SQL_AGG, SQL_OPS, SQL_ORDERBY_OPS, SQL_DISTINCT_OP
import numpy as np
import torch
from itertools import chain
from utils.utils import pad, text2int
from nltk.tokenize import word_tokenize
import re

def zero_pad(sequences):

    """
    Args:
        sequences [seq_len]: sequences to be padded with zeros
    Returns:
        padded_seqs [seq_len, max_len]: sequences of equal length but padded with zeros after original length is reached
        lengths [seq_len]: list of sequence lengths 

    """
    lengths = [len(seq) for seq in sequences]
    padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
    for i, seq in enumerate(sequences):
        end = lengths[i]
        padded_seqs[i, :end] = seq[:end]
    return padded_seqs, lengths

class SpiderDataset(Dataset):
    """
    Wrapper around the Spider dataset, that automatically removes selected keywords and caches the sql, question,
    database and histoy in memory as well as generating the different datasets for each module.

    """
    def __init__(self, data_path, tables_path, exclude_keywords=[], debug=True, language='en'):
        """
        Args:
            data_path (string): file path of the data json file with questions
            tables_path (string): file path of the tables json file with db schema
        """		
        directory=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
		
        self.exclude_keywords = exclude_keywords
        self.data = []

        data = json.load(open(directory + '/' + data_path, 'r', encoding="utf8"))

        # Handle excluded keywords, by removing them, and logging how many of each type was found
        exclude_keywords_counts = {key: 0 for key in exclude_keywords}
        for d in data:
            keywords = [keyword for keyword in exclude_keywords if str.upper(keyword) in str.upper(d['query'])]
            if keywords:
                for keyword in keywords:
                    exclude_keywords_counts[keyword] += 1
            else:
                self.data += [d]
        if debug:
            for keyword in exclude_keywords_counts:
                print(f"Found {exclude_keywords_counts[keyword]} queries with excluded keyword {keyword}")
            print(f"Total number of removed queries = {len(data) - len(self.data)} / {len(data)}")

        tables = json.load(open(directory + '/' + tables_path, 'r'))
        # change the key of the dictionary to the db_id
        self.tables = {}
        for table in tables:
            db_id = table['db_id']
            self.tables[db_id] = table

        self.samples = []

        p = re.compile(r"(?:(?<!\w)'((?:.|\n)+?'?)'(?!\w))")
        # Cache the preprocessing in memory
        failed = 0
        for i in range(len(self.data)):
            try:
                example = self.data[i]
                db_id = example['db_id']
                db = DataBase(self.tables[db_id])

                sql = SQLStatement(query=example['query'], database=db)
                # TODO: include other languages
                # Replace ' with " to split the words correctly
                question = re.sub(p, u"\"\g<1>\"", example['question'][language])
                
                history = sql.generate_history()

                sample = {'sql': sql, 'question': question, 'db': db, 'history': history}
                self.samples += [sample]
            except:
                failed += 1
        if failed > 0:
            print(f"{failed}/{len(self.data)} queries could not be loaded")

    def __len__(self):
        return len(self.data)

    def get_common_data(self, sample):
        db = sample['db']
        sql = sample['sql']
        # Get a list of all columns in the database
        columns_all = db.to_list()
        #split connected words like 'address_id' into address, id
        # TODO: Should we do this in the database object 
        columns_all_splitted = []
        for i, column in enumerate(columns_all):
            columns_tmp = []
            for word in column:
                columns_tmp.extend(word.split('_'))
            
            columns_all_splitted += [columns_tmp]

        question = sample['question']
        return db, sql, columns_all_splitted, columns_all, question

    def __getitem__(self, idx):
        return self.samples[idx]

    def generate_keyword_dataset(self):
        dataset = []
        for sample in self.samples:
            db = sample['db']
            sql = sample['sql']
            # Convert keywords to indecies

            keywords_idx = [SQL_KEYWORDS.index(keyword) for keyword in sql.keywords]
            keywords_onehot = np.zeros(len(SQL_KEYWORDS))
            keywords_onehot[keywords_idx] = 1
            num_keywords = len(keywords_idx)

            history = sample['history']['keyword']
            question = sample['question']

            dataset.append({'num_keywords': num_keywords, 'keywords': keywords_onehot, 'question': question, 'history': history, 'db': db, 'sql': sql})

        return ModularDataset(dataset, name='KeyWord')

    def generate_andor_dataset(self):
        dataset = []
        for sample in self.samples:
            db = sample['db']
            sql = sample['sql']
            question = sample['question']

            for andor, history in zip(sql.and_ors, sample['history']['andor']):
                andor_idx = SQL_COND_OPS.index(andor)
                
                #We use andor_idx as a list since this is the way our collate function expects input

                dataset.append({'andor': [andor_idx], 'question': question, 'history': history, 'db': db, 'sql': sql})

        return ModularDataset(dataset, name='AndOr')

    def generate_column_dataset(self):
        dataset = []
        for sample in self.samples:

            db, sql, columns_all_splitted, columns_all, question = self.get_common_data(sample)

            #We predict the columns for each group as a separate prediction, so split them up into different data points    
            #In order to match with the history, just take the nonempty groups
            groups = [group for group in (sql.COLS, sql.WHERE, sql.GROUPBY, sql.HAVING, sql.ORDERBY) if group]
            for columns, history in zip(groups, sample['history']['col']):
                
                #Get the index of the target column, from the lists of all columns in the database
                columns_idx = [columns_all.index(col.column.to_list()) for col in columns]
                #Convert to onehot encoding
                columns_onehot = np.zeros(len(columns_all))
                for i in columns_idx:
                    columns_onehot[i]+= 1
                num_columns = len(columns_idx)

                dataset.append({'columns_all':columns_all_splitted, 'num_columns': [num_columns], 'columns': columns_onehot, 'question': question, 'history': history, 'db': db, 'sql': sql})

        return ModularDataset(dataset, name='Column')

    def generate_agg_dataset(self):
        dataset = []
        for sample in self.samples:
            
            db, sql, columns_all_splitted, columns_all, question = self.get_common_data(sample)
            
            #In order to match with the history, just take the nonempty columns
            columns = [group for group in chain(sql.COLS, sql.HAVING, sql.ORDERBY) if group]
            for column, history in zip(columns, sample['history']['agg']):

                #Get the index of the target column, from the lists of all columns in the database
                column_idx = columns_all.index(column.column.to_list()) 
                #Get index of the aggregator
                agg_idx = SQL_AGG.index(column.agg)

                dataset.append({'columns_all':columns_all_splitted, 'column_idx': column_idx, 'agg': [agg_idx], 'question': question, 'history': history, 'db': db, 'sql': sql})

        return ModularDataset(dataset, name='Aggregate')

    def generate_distinct_dataset(self):
        dataset = []
        for sample in self.samples:
            
            db, sql, columns_all_splitted, columns_all, question = self.get_common_data(sample)
            
            #In order to match with the history, just take the nonempty columns
            columns = [group for group in chain(sql.COLS) if group]
            for column, history in zip(columns, sample['history']['distinct']):

                #Get the index of the target column, from the lists of all columns in the database
                column_idx = columns_all.index(column.column.to_list()) 
                #Get index of the aggregator
                dist_idx = SQL_DISTINCT_OP.index(column.distinct)

                dataset.append({'columns_all':columns_all_splitted, 'column_idx': column_idx, 'distinct': [dist_idx], 'question': question, 'history': history, 'db': db, 'sql': sql})

        return ModularDataset(dataset, name='Distinct')
    
    def generate_op_dataset(self):
        dataset = []
        for sample in self.samples:
            
            db, sql, columns_all_splitted, columns_all, question = self.get_common_data(sample)
            
            #In order to match with the history, just take the nonempty columns
            conditions = [group for group in chain(sql.WHERE, sql.HAVING) if group]
            for condition, history in zip(conditions, sample['history']['op']):

                #Get the index of the target column, from the lists of all columns in the database
                column_idx = columns_all.index(condition.column.to_list()) 
                #Get index of the aggregator
                op_idx = SQL_OPS.index(condition.op)

                dataset.append({'columns_all':columns_all_splitted, 'column_idx': column_idx, 'op': [op_idx], 'question': question, 'history': history, 'db': db, 'sql': sql})

        return ModularDataset(dataset, name='OP')

    def generate_having_dataset(self):
        dataset = []
        for sample in self.samples:
            db = sample['db']
            sql = sample['sql']
            
            columns_all = db.to_list()
            #split connected words like 'address_id' into address, id
            # TODO: Should we do this in the database object 
            columns_all_splitted = []
            for i, column in enumerate(columns_all):
                columns_tmp = []
                for word in column:
                    columns_tmp.extend(word.split('_'))
                
                columns_all_splitted += [columns_tmp]

            #TODO: instead of just picking the first column, pick something more saying
            column_idx = columns_all.index(sql.COLS[0].column.to_list()) 

            #Check if sql has having clause
            having = int(bool(sql.HAVING))
            history = sample['history']['having'][0]
            question = sample['question']

            dataset.append({'having': [having], 'column_idx':column_idx, 'columns_all':columns_all_splitted, 'question': question, 'history': history, 'db': db, 'sql': sql})

        return ModularDataset(dataset, name='Having')

    def generate_desasc_dataset(self):
        dataset = []
        for sample in self.samples:
            
            db, sql, columns_all_splitted, columns_all, question = self.get_common_data(sample)           

            for orderby, orderby_op, history in zip(sql.ORDERBY, sql.ORDERBY_OP, sample['history']['decasc']):

                #Get the index of the target column, from the lists of all columns in the database
                column_idx = columns_all.index(orderby.column.to_list()) 
                desasc = SQL_ORDERBY_OPS.index(orderby_op)

                dataset.append({'columns_all':columns_all_splitted, 'column_idx': column_idx, 'desasc': [desasc], 'question': question, 'history': history, 'db': db, 'sql': sql})

        return ModularDataset(dataset, name='Desasc')

    def generate_value_dataset(self):
        dataset = []
        for sample in self.samples:
            
            db, sql, columns_all_splitted, columns_all, question = self.get_common_data(sample)
            
            # In order to match with the history, just take the nonempty columns
            conditions = [group for group in chain(sql.WHERE, sql.HAVING) if group]
            for condition, history in zip(conditions, sample['history']['value']):
                
                 
                # Get the index of the target column, from the lists of all columns in the database
                column_idx = columns_all.index(condition.column.to_list())

                # Get target value
                value = word_tokenize(str.lower(condition.value))
                
                if value:
                    start_token = value[0]
                    num_tokens = len(value)
                #TODO the value might be '' which doesn't work with the tokenizer at the moment
                else:
                    start_token = ''
                    num_tokens = 1
                
                #Convert to onehot encoding
                tokens=word_tokenize(text2int(str.lower(sample['question'])))
                values_onehot = np.zeros(len(tokens))
                try:
                    values_onehot[tokens.index(start_token)] = 1
                except:
                    pass
                

                dataset.append({'columns_all':columns_all_splitted, 'column_idx': column_idx, 'value': values_onehot, 'num_tokens':[num_tokens], 'question': question, 'history': history, 'db': db, 'sql': sql})

        return ModularDataset(dataset, name='Value')

    def generate_limitvalue_dataset(self):
        dataset = []
        for sample in self.samples:
            
            db, sql, columns_all_splitted, columns_all, question = self.get_common_data(sample)           

            for orderby, orderby_op, limitvalue, history in zip(sql.ORDERBY, sql.ORDERBY_OP, sql.LIMIT_VALUE, sample['history']['decasc']):

                # Get the index of the target column, from the lists of all columns in the database
                column_idx = columns_all.index(orderby.column.to_list()) 
                desasc = SQL_ORDERBY_OPS.index(orderby_op)
                limitvalue = int(limitvalue)

                dataset.append({'columns_all':columns_all_splitted, 'column_idx': column_idx, 'desasc': [desasc], 'limitvalue': [limitvalue], 'question': question, 'history': history, 'db': db, 'sql': sql})

        return ModularDataset(dataset, name='LimitValue')

class ModularDataset(Dataset):

    def __init__(self, data, name=''):
        super(ModularDataset, self).__init__()
        self.data = data
        self.name = name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __repr__(self):
        return f"{self.name}Dataset"


def try_tensor_collate_fn(batch):
    """
    Try to batch the inputs and convert them into tensors, but fall back to lists.
    In practice this means that only numeric input will be tensors, and meta data like sql, db stuff and questions are kept as lists
    """
    output = {}

    for example in batch:
        for key in example:
            if key in output:
                output[key] += [example[key]]
            else:
                output[key] = [example[key]]

    for key in output:
        try:
            output[key] = torch.tensor(pad(output[key])[0])
        except:
            pass
    return output


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    spider = SpiderDataset(data_path='data/'+'train.json', tables_path='/data/'+'tables.json', exclude_keywords=['-', ' / ', ' + '])
    dat = spider.generate_value_dataset()
    # spider[0]

    dl = DataLoader(dat, batch_size=2, collate_fn=try_tensor_collate_fn)

    b = next(iter(dl))
    a=1
