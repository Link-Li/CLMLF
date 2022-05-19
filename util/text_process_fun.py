# -*- encoding: utf-8 -*-
'''
@File    :   text_process_fun.py
@Time    :   2020/06/25 11:36:40
@Version :   1.0
'''

# here put the code
import re
from nltk.corpus import wordnet


# import nltk
# nltk.download('wordnet')


class AbbreviationReduction():

    def __init__(self):
        replacement_patterns = [
            (r'won\'t', 'will not'),
            (r'can\'t', 'can not'),
            (r'i\'m', 'i am'),
            (r'ain\'t', 'is not'),
            (r'(\w+)\'ll', '\g<1> will'),
            (r'(\w+)n\'t', '\g<1> not'),
            (r'(\w+)\'ve', '\g<1> have'),
            (r'(\w+)\'s', '\g<1> is'),
            (r'(\w+)\'re', '\g<1> are'),
            (r'(\w+)\'d', '\g<1> would')]

        self.patterns = [(re.compile(regex, re.I), repl)
                         for (regex, repl) in replacement_patterns]

    def replace(self, text):
        s = text
        for (pattern, repl) in self.patterns:
            (s, count) = re.subn(pattern, repl, s)
        return s


class RepeatReplacer():

    def __init__(self):
        # 重复三个单词就认为是重复了, 这里不区分大小写进行匹配
        self.repeat_reg = re.compile(r'(\w*)(\w)\2(\w*)', re.I)
        self.repl = r'\1\2\3'

    def replace(self, word):
        if wordnet.synsets(word):  # 判断当前字符串是否是单词
            return word
        repl_word = self.repeat_reg.sub(self.repl, word)
        if repl_word != word:
            return self.replace(repl_word)
        else:
            return repl_word


class UrlProcess():
    def __init__(self):
        # 去除#开头的
        self.hashtag_pattern = re.compile('.?#[a-zA-Z0-9_\.]+', re.I)
        # self.hashtag_pattern = re.compile('#[a-zA-Z0-9]+')
        # 去除@开头的
        self.at_pattern = re.compile('.?@[a-zA-Z0-9_\.]+', re.I)
        # self.at_pattern = re.compile('@[a-zA-Z0-9]+')
        # 去除网址的
        self.http_pattern = re.compile("(http|ftp|https)://[a-zA-Z0-9\./]+|www\.(\w+\.)+\S*/", re.I)
        # self.http_pattern = re.compile(
        #     "((http|ftp|https)://)(([a-zA-Z0-9\._-]+\.[a-zA-Z]{2,6})|([0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}))(:[0-9]{1,4})*(/[a-zA-Z0-9\&%_\./-~-]*)?")

    def replace(self, text):
        text = re.sub(self.hashtag_pattern, '', text)
        text = re.sub(self.at_pattern, '', text)
        text = re.sub(self.http_pattern, '', text)
        return text


class TextProcess_0():
    def __init__(self):
        pass

    def process(self, text):
        return text


class TextProcess_1():
    def __init__(self):
        self.abbreviation_reduction = AbbreviationReduction()
        self.repeat_replacer = RepeatReplacer()

    def process(self, text):
        text = self.abbreviation_reduction.replace(text)
        text = self.repeat_replacer.replace(text)
        return text


class TextProcess_2():
    def __init__(self):
        self.url_replacer = UrlProcess()

    def process(self, text):
        text = self.url_replacer.replace(text)
        return text


class TextProcess_3():
    def __init__(self):
        self.abbreviation_reduction = AbbreviationReduction()
        self.repeat_replacer = RepeatReplacer()
        self.url_replacer = UrlProcess()

    def process(self, text):
        text = self.abbreviation_reduction.replace(text)
        text = self.repeat_replacer.replace(text)
        text = self.url_replacer.replace(text)
        return text

# ar = AbbreviationReduction()
# # text = "QUIZ: Which ‘Sex and the City' Gal's Style of Internalized Misogyny Are You? Entertainment - Jan 12  2017 By: Liz Arcury"
# text = "When your bff's ex tried to walk back into her life and you aren't having any of it."
# a = ar.replace(text)
# print(a)

# replacer = RepeatReplacer()
# test1 = replacer.replace("TIL Andrew Lincoln was originally cast as Marlin from Finding Nemo Caaarl!!! Coooraaal!!! The Walking Coral")
# test2 = replacer.replace('Oooh! PIN Me. Pin ME!')
# test3 = replacer.replace('When your crush passes by and you wanna see how he/she is looking lo looking good ooo you smexy @baemillssey hot P.E.R.F.E.CT ot yeahp you are'.lower())
# print(test1)
# print(test2)
# print(test3)

# import n
