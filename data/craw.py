from urllib2 import *
from bs4 import BeautifulSoup
import os
x=0
flag = 0

dict = {}

dict_fail = {}


def craw(id):
    try:

        if dict_fail.has_key(id):
            return False
        if id[0] != '/':
            id = '/' + id

        print 'Crawing %s' % (id)
        html = urlopen("https://www.freebase.com" + id)
        soup = BeautifulSoup(html, "html.parser")
        content = soup.select('#max-blurb')[0].contents[0]
        content = content.string
        str_content = content.text.encode('utf-8').decode('ascii', 'ignore')
        content  = str_content.encode('ascii')

        str_content = str(content)
        # print(str_content)
        str_content = str_content.replace('\n', ' ')
        str_content = str_content.replace('\t',' ')
        # print(str_content)
        words = str_content.split(' ')
        # print words
        sentence = ''
        word_count = 0
        for w in words:
            if w == '':
                pass
            else:
                sentence += w + ' '
                word_count += 1
        sentence = sentence[:-1]
        # print sentence
        if word_count <= 3:
            dict_fail[id] = True
            return False
        else:
            dict[id] = sentence
            with open('fb_des.txt', 'a') as des:
                des.write('%s\t%s\n' % (id, sentence))
            return True
    except UnicodeEncodeError:
        print('\t\tUnicode Error!%s' % (id))
        dict_fail[id] = True
        return False
    except:
        print('\t\tError!%s' % (id))
        dict_fail[id] = True
        return False


# os.chdir('/Users/jcxu/Projects/keras/data')
craw('/m/0hk18')


with open('new_fb_des.txt', 'r') as des:
    lines = des.readlines()
    for l in lines:
        id, sentence = l.split('\t')
        if dict.has_key(id):
            print('Overlap!')
        else:
            dict[str(id)] = sentence

print 'load old over'

list_to_be = []
dict_to_be = {}

with open('FB40k.txt') as old_train:
    lines = old_train.readlines()
    for i in xrange(len(lines)):
        l = lines[i]
        if len(l.split('\t')) == 2:
            ht, r = l.split('\t')
            h, t = ht.split(' ')
            h = '/' + h
            t = '/' + t
        elif len(l.split('\t')) == 3:
            h, t, r = l.split('\t')
            # h,t = ht.split(' ')
            h = '/' + h
            t = '/' + t
        else:
            raise NotImplementedError
        h_flag, t_flag = True, True
        if dict.has_key(h) or dict_to_be.has_key(h):
            pass
        else:
            list_to_be.append(h)
            dict_to_be[h] = True

        if dict.has_key(t) or dict_to_be.has_key(t):
            pass
        else:
            list_to_be.append(t)
            dict_to_be[t] = True

new_train = []
# Format:
# [['/m/027rn', '/m/06cx9', '/location/country/form_of_government'], ['/m/017dcd', '/m/06v8s0', '/tv/tv_program/regular_cast./tv/regular_tv_appearance/actor']]
import numpy
import threading


class MyThread(threading.Thread):
    def __init__(self, i):
        super(MyThread, self).__init__()
        self.i = i

    def run(self):
        for i in xrange(len(list_to_be)):
            if i % 8 == self.i:
                craw(list_to_be[i])


def test():

    t = MyThread(x)
    t.start()


test()

with open('FB40k.txt') as old_train:
    lines = old_train.readlines()
    for l in lines:
        h, t, r = l.split('\t')
        r = r[:-1]
        h_flag, t_flag = True, True
        if dict.has_key(h):
            pass
        else:
            h_flag = craw(h)

        if dict.has_key(t):
            pass
        else:
            t_flag = craw(t)

        if h_flag == True and t_flag == True:
            new_train.append([h, t, r])

print 'old data lenth\n335350'
print 'new data lenth'
print(len(new_train))
import cPickle as pkl

pkl.dump([dict, new_train], open('fb40k_des.pkl', 'wb'))
