# -*- coding: utf-8 -*-
u"""Official Evaluator for WikiTableQuestions Dataset
There are 3 value types
1. String (unicode)
2. Number (float)
3. Date (a struct with 3 fields: year, month, and date)
   Some fields (but not all) can be left unspecified. However, if only the year
   is specified, the date is automatically converted into a number.
Target denotation = a set of items
- Each item T is a raw unicode string from Mechanical Turk
- If T can be converted to a number or date (via Stanford CoreNLP), the
    converted value (number T_N or date T_D) is precomputed
Predicted denotation = a set of items
- Each item P is a string, a number, or a date
- If P is read from a text file, assume the following
  - A string that can be converted into a number (float) is converted into a
    number
  - A string of the form "yyyy-mm-dd" is converted into a date. Unspecified
    fields can be marked as "xx". For example, "xx-01-02" represents the date
    January 2nd of an unknown year.
  - Otherwise, it is kept as a string
The predicted denotation is correct if
1. The sizes of the target denotation and the predicted denotation are equal
2. Each item in the target denotation matches an item in the predicted
    denotation
A target item T matches a predicted item P if one of the following is true:
1. normalize(raw string of T) and normalize(string form of P) are identical.
   The normalize method performs the following normalizations on strings:
   - Remove diacritics (é → e)
   - Convert smart quotes (‘’´`“”) and dashes (‐‑‒–—−) into ASCII ones
   - Remove citations (trailing •♦†‡*#+ or [...])
   - Remove details in parenthesis (trailing (...))
   - Remove outermost quotation marks
   - Remove trailing period (.)
   - Convert to lowercase
   - Collapse multiple whitespaces and strip outermost whitespaces
2. T can be interpreted as a number T_N, P is a number, and P = T_N
3. T can be interpreted as a date T_D, P is a date, and P = T_D
   (exact match on all fields; e.g., xx-01-12 and 1990-01-12 do not match)
"""
__version__ = '1.0.2'

import sys
import os
import re
import unicodedata
import requests
import json
from codecs import open
from math import isnan, isinf
from abc import ABCMeta, abstractmethod
import sqlite3
from stanfordnlp.server import CoreNLPClient
# import func_timeout

################ String Normalization ################

def normalize(x):
    if not isinstance(x, str):
        x = x.decode('utf8', errors='ignore')
    # Remove diacritics
    x = ''.join(c for c in unicodedata.normalize('NFKD', x)
                if unicodedata.category(c) != 'Mn')
    # Normalize quotes and dashes
    x = re.sub(r"[‘’´`]", "'", x)
    x = re.sub(r"[“”]", "\"", x)
    x = re.sub(r"[‐‑‒–—−]", "-", x)
    while True:
        old_x = x
        # Remove citations
        x = re.sub(r"((?<!^)\[[^\]]*\]|\[\d+\]|[•♦†‡*#+])*$", "", x.strip())
        # Remove details in parenthesis
        x = re.sub(r"(?<!^)( \([^)]*\))*$", "", x.strip())
        # Remove outermost quotation mark
        x = re.sub(r'^"([^"]*)"$', r'\1', x.strip())
        if x == old_x:
            break
    # Remove final '.'
    if x and x[-1] == '.':
        x = x[:-1]
    # Collapse whitespaces and convert to lower case
    x = re.sub(r'\s+', ' ', x, flags=re.U).lower().strip()
    return x


################ Value Types ################

class Value(object):
    __metaclass__ = ABCMeta

    # Should be populated with the normalized string
    _normalized = None

    @abstractmethod
    def match(self, other):
        """Return True if the value matches the other value.
        Args:
            other (Value)
        Returns:
            a boolean
        """
        pass

    @property
    def normalized(self):
        return self._normalized


class StringValue(Value):

    def __init__(self, content):
        assert isinstance(content, str)
        self._normalized = normalize(content)
        self._hash = hash(self._normalized)

    def __eq__(self, other):
        return isinstance(other, StringValue) and self.normalized == other.normalized

    def __hash__(self):
        return self._hash

    def __str__(self):
        return 'S' +  str([self.normalized])
    __repr__ = __str__

    def match(self, other):
        assert isinstance(other, Value)
        return self.normalized == other.normalized


class NumberValue(Value):

    def __init__(self, amount, original_string=None):
        assert isinstance(amount, (int, float))
        if abs(amount - round(amount)) < 1e-6:
            self._amount = int(amount)
        else:
            self._amount = float(amount)
        if not original_string:
            self._normalized = unicode(self._amount)
        else:
            self._normalized = normalize(original_string)
        self._hash = hash(self._amount)

    @property
    def amount(self):
        return self._amount

    def __eq__(self, other):
        return isinstance(other, NumberValue) and self.amount == other.amount

    def __hash__(self):
        return self._hash

    def __str__(self):
        return ('N(%f)' % self.amount) + str([self.normalized])
    __repr__ = __str__

    def match(self, other):
        assert isinstance(other, Value)
        if self.normalized == other.normalized:
            return True
        if isinstance(other, NumberValue):
            return abs(self.amount - other.amount) < 1e-6
        return False

    @staticmethod
    def parse(text):
        """Try to parse into a number.
        Return:
            the number (int or float) if successful; otherwise None.
        """
        try:
            return int(text)
        except:
            try:
                amount = float(text)
                assert not isnan(amount) and not isinf(amount)
                return amount
            except:
                return None


class DateValue(Value):

    def __init__(self, year, month, day, original_string=None):
        """Create a new DateValue. Placeholders are marked as -1."""
        assert isinstance(year, int)
        assert isinstance(month, int) and (month == -1 or 1 <= month <= 12)
        assert isinstance(day, int) and (day == -1 or 1 <= day <= 31)
        assert not (year == month == day == -1)
        self._year = year
        self._month = month
        self._day = day
        if not original_string:
            self._normalized = '{}-{}-{}'.format(
                year if year != -1 else 'xx',
                month if month != -1 else 'xx',
                day if day != '-1' else 'xx')
        else:
            self._normalized = normalize(original_string)
        self._hash = hash((self._year, self._month, self._day))

    @property
    def ymd(self):
        return (self._year, self._month, self._day)

    def __eq__(self, other):
        return isinstance(other, DateValue) and self.ymd == other.ymd

    def __hash__(self):
        return self._hash

    def __str__(self):
        return (('D(%d,%d,%d)' % (self._year, self._month, self._day))
                + str([self._normalized]))
    __repr__ = __str__

    def match(self, other):
        assert isinstance(other, Value)
        if self.normalized == other.normalized:
            return True
        if isinstance(other, DateValue):
            return self.ymd == other.ymd
        return False

    @staticmethod
    def parse(text):
        """Try to parse into a date.
        Return:
            tuple (year, month, date) if successful; otherwise None.
        """
        try:
            ymd = text.lower().split('-')
            assert len(ymd) == 3
            year = -1 if ymd[0] in ('xx', 'xxxx') else int(ymd[0])
            month = -1 if ymd[1] == 'xx' else int(ymd[1])
            day = -1 if ymd[2] == 'xx' else int(ymd[2])
            assert not (year == month == day == -1)
            assert month == -1 or 1 <= month <= 12
            assert day == -1 or 1 <= day <= 31
            return (year, month, day)
        except:
            return None


################ Value Instantiation ################

def to_value(original_string, corenlp_value=None):
    """Convert the string to Value object.
    Args:
        original_string (basestring): Original string
        corenlp_value (basestring): Optional value returned from CoreNLP
    Returns:
        Value
    """
    if isinstance(original_string, Value):
        # Already a Value
        return original_string
    if not corenlp_value:
        corenlp_value = original_string
    # Number?
    amount = NumberValue.parse(corenlp_value)
    if amount is not None:
        return NumberValue(amount, original_string)
    # Date?
    ymd = DateValue.parse(corenlp_value)
    if ymd is not None:
        if ymd[1] == ymd[2] == -1:
            return NumberValue(ymd[0], original_string)
        else:
            return DateValue(ymd[0], ymd[1], ymd[2], original_string)
    # String.
    return StringValue(original_string)

def to_value_list(original_strings, corenlp_values=None):
    """Convert a list of strings to a list of Values
    Args:
        original_strings (list[basestring])
        corenlp_values (list[basestring or None])
    Returns:
        list[Value]
    """
    assert isinstance(original_strings, (list, tuple, set))
    if corenlp_values is not None:
        assert isinstance(corenlp_values, (list, tuple, set))
        assert len(original_strings) == len(corenlp_values)
        return list(set(to_value(x, y) for (x, y)
                in zip(original_strings, corenlp_values)))
    else:
        return list(set(to_value(x) for x in original_strings))


################ Check the Predicted Denotations ################

def check_denotation(target_values, predicted_values):
    """Return True if the predicted denotation is correct.

    Args:
        target_values (list[Value])
        predicted_values (list[Value])
    Returns:
        bool
    """
    # Check size
    if len(target_values) != len(predicted_values):
        return False
    # Check items
    for target in target_values:
        if not any(target.match(pred) for pred in predicted_values):
            return False
    return True


################ Batch Mode ################

def tsv_unescape(x):
    """Unescape strings in the TSV file.
    Escaped characters include:
        newline (0x10) -> backslash + n
        vertical bar (0x7C) -> backslash + p
        backslash (0x5C) -> backslash + backslash
    Args:
        x (str or unicode)
    Returns:
        a unicode
    """
    return x.replace(r'\n', '\n').replace(r'\p', '|').replace('\\\\', '\\')

def tsv_unescape_list(x):
    """Unescape a list in the TSV file.
    List items are joined with vertical bars (0x5C)
    Args:
        x (str or unicode)
    Returns:
        a list of unicodes
    """
    return [tsv_unescape(y) for y in x.split('|')]

def check_quote_format(str):
    single_start = False
    double_start = False
    for i in range(len(str)):
        # print('single ', single_start, 'double ', double_start)
        if str[i] == '"' and single_start==False:
            if double_start==False:
                double_start = True
                continue
            else:
                double_start=False
                continue
        if str[i] == '\'' and double_start==False:
            if single_start==False:
                single_start = True
                continue
            else:
                single_start=False
                continue
        if i == len(str)-1 and (single_start==True or double_start==True):
            return False
    return True



def make_query(sql, is_list, c, pred, replace=True):
    sql = requests.get(
        "http://localhost:3000/", 
        json={"sql": sql, "is_list": is_list}).json()
    c.execute(sql)
    answer_list = list()
    for result, in c:
        result = str(result)
        answer_list.append(result)
        
    pred['nl'] = pred['nl'].lower()
    if replace:
        if any(item in pred['nl'] for item in ['how many', 
                                               'what is the number', 
                                               'what was the number',
                                               'what are the total number',
                                               'what was the total number',
                                               'what rank']):
            pass
        elif any(item in pred['nl'] for item in ['more/less', 'more or less']) and answer_list[0] in ['0','1']:
            replace_dict = {'0':'less', '1':'more'}
            answer_list = [replace_dict[answer_list[0]]]
        elif any(item in pred['nl'] for item in ['above or below', 'above/below']) and answer_list[0] in ['0','1']:
            replace_dict = {'0':'below', '1':'above'}
            answer_list = [replace_dict[answer_list[0]]]
        elif any(pred['nl'].startswith(prefix) for prefix in ['is', 'was', 'does', 'do', 'did', 'were']) and answer_list[0] in ['0','1']:
            replace_dict = {'0':'no', '1':'yes'}
            answer_list = [replace_dict[answer_list[0]]]
        elif any(item in pred['nl'] for item in ['month']) and answer_list[0] in [str(n) for n in range(1,13)]:
            replace_dict = {
                '1': 'January',
                '2': 'February',
                '3': 'March',
                '4': 'April',
                '5': 'May',
                '6': 'June',
                '7': 'July',
                '8': 'August',
                '9': 'September',
                '10': 'October',
                '11': 'November',
                '12': 'December'
            }
            answer_list = [replace_dict[answer_list[0]]]

    predicted_values = to_value_list(answer_list)
    return predicted_values, answer_list

def eval_tag_match(pred, ex_id, target_values_map, separator='|'):
    target_values = target_values_map[ex_id]
    predicted_values = to_value_list(pred.split(separator))
    correct = check_denotation(target_values, predicted_values)
    return correct

class Evaluator:
    def __init__(self):
        self.target_values_map = {}
        self.tagged_dataset_path = './data/squall/tables/tagged'
        for filename in os.listdir(self.tagged_dataset_path):
            filename = os.path.join(self.tagged_dataset_path, filename)
            print(sys.stderr, 'Reading dataset from', filename)
            with open(filename, 'r', 'utf8') as fin:
                header = fin.readline().rstrip('\n').split('\t')
                for line in fin:
                    stuff = dict(zip(header, line.rstrip('\n').split('\t')))
                    ex_id = stuff['id']
                    original_strings = tsv_unescape_list(stuff['targetValue'])
                    canon_strings = tsv_unescape_list(stuff['targetCanon'])
                    self.target_values_map[ex_id] = to_value_list(
                            original_strings, canon_strings)

    def evaluate_text_to_sql(self, predictions):
        correct_flag = []
        targets = []
        predicted = []
        buffer = {}
        for pred in predictions:
            table_id = pred['table_id']
            if table_id not in buffer:
                with open(pred['json_path'], "r") as f:
                    table_json = json.load(f)
                buffer[table_id] = table_json["is_list"]

            db_file = pred['db_path']
            connection = sqlite3.connect(db_file)
            c = connection.cursor()
            results = pred['result']
            for result in results:
                answer_list = list()
                ex_id = result['id']
                targets.append(self.target_values_map[ex_id])
                sql = result['sql']
                if not check_quote_format(sql):
                    sql = "select"
                try:
                    is_list = buffer[table_id]
                    predicted_values, answer_list = make_query(sql, is_list, c, pred, replace=True)
                except Exception as e:
                    predicted_values = list()

                if ex_id not in self.target_values_map:
                    print('WARNING: Example ID "%s" not found' % ex_id)
                    raise ValueError
                    # correct_flag.append(None)
                else:
                    target_values = self.target_values_map[ex_id]
                    correct = check_denotation(target_values, predicted_values)
                    correct_flag.append(correct)
                
                predicted.append('|'.join([str(x) for x in answer_list]))

        assert len(correct_flag)==len(predictions)
        return correct_flag, predicted
        
    def evaluate_tableqa(self, predictions):
        correct_flag = []
        for prediction in predictions:
            pred = prediction['pred']
            ex_id = prediction['nt']
            correct_flag.append(eval_tag_match(pred, ex_id, self.target_values_map, separator='|'))
        return correct_flag
