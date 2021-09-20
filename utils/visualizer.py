import os
import pdb
import time
import sys
import socket
import torch
from tensorboardX import SummaryWriter
from datetime import datetime


def get_summary_writer(log_dir,name=None):
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if name is None:
        log_dir = os.path.join(log_dir, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    else:
        log_dir = os.path.join(log_dir, name)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    writer = SummaryWriter(log_dir)
    return writer


class smoothMeters(object):
    def __init__(self, dic=None, total_num=None):
        self.dic = dic or {}
        # self.total_num = total_num
        self.dic_last = dic or {}

    def update(self, new_dic):
        for key in new_dic:
            if not key in self.dic:
                self.dic[key] = new_dic[key]
                self.dic_last[key] = new_dic[key]
            else:
                self.dic_last[key] = self.dic[key]
                self.dic[key] = new_dic[key]*0.1+self.dic_last[key]*0.9
        # self.total_num += 1

    def __getitem__(self, key):
        return self.dic[key]

    def __str__(self):
        keys = sorted(self.keys())
        res = ''
        for key in keys:
            res += (key + ': %.4f' % self[key] + ' | ')
        return res

    def keys(self):
        return self.dic.keys()

class avgMeters(object):
    def __init__(self, dic=None, total_num=None):
        self.dic = dic or {}
        # self.total_num = total_num
        self.total_num = total_num or {}

    def update(self, new_dic):
        for key in new_dic:
            if not key in self.dic:
                self.dic[key] = new_dic[key]
                self.total_num[key] = 1
            else:
                self.dic[key] += new_dic[key]
                self.total_num[key] += 1
        # self.total_num += 1

    def __getitem__(self, key):
        return self.dic[key] / self.total_num[key]

    def __str__(self):
        keys = sorted(self.keys())
        res = ''
        for key in keys:
            res += (key + ': %.4f' % self[key] + ' | ')
        return res

    def keys(self):
        return self.dic.keys()


def write_loss(writer, prefix, avg_meters, iteration):
    for key in avg_meters.keys():
        meter = avg_meters[key]
        writer.add_scalar(
            os.path.join(prefix, key), meter, iteration)


# def error_handler(func):
#     def wrapper(*args, **kwargs):
#         try:
#             return func(*args, **kwargs)
#         except:
#             pass
#     return wrapper #返回函数名


def decorator(func):
    def inside(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except :
            print('数据内部错误')
    return inside

# def error_handler(a_func):
#     def wrapTheFunction(*args, **kwargs):
#         try:
#             a_func(*args, **kwargs)
#         except:
#             import ipdb;
#             ipdb.set_trace()
#             pass
#
#     return wrapTheFunction

@decorator
def write_image(writer: SummaryWriter, prefix, image_name: str, img, epoch):
    """
        :param writer:
        :param prefix:
        :param image_name:
        :param img: a pytorch image tensor [C, H, W]
        :param iteration:
        :return:
    """
    if img==None:
        return
    img = torch.clamp(img,0,1)
    writer.add_image(
        os.path.join(prefix, image_name), img, epoch, dataformats='CHW')


"""progress bar"""

class bar(object):
    def __init__(self):
        super(bar,self).__init__()
        _, term_width = os.popen('stty size', 'r').read().split()
        self.term_width = int(term_width)
        self.TOTAL_BAR_LENGTH = 60
        self.last_time = time.time()
        self.avg_time =0
        self.begin_time=self.last_time
    def progress_bar(self,current, total, msg=None):
        cur_len = int(self.TOTAL_BAR_LENGTH*current/total)
        rest_len = int(self.TOTAL_BAR_LENGTH - cur_len) - 1

        sys.stdout.write(' [')
        for i in range(cur_len):
            sys.stdout.write('=')
        sys.stdout.write('>')
        for i in range(rest_len):
            sys.stdout.write('.')
        sys.stdout.write(']')

        cur_time = time.time()
        step_time = cur_time - self.last_time
        self.last_time = cur_time
        tot_time = cur_time - self.begin_time

        self.avg_time = self.avg_time * 0.9 + step_time*0.1
        if current == 1:
            self.avg_time=step_time
        # pdb.set_trace()


        L = []
        L.append('  Step: %s' % format_time(step_time))
        L.append(' | Tot: %s' % format_time(tot_time))
        L.append(' | ETA: %s' % format_time((total-current-1)*self.avg_time))
        if msg:
            L.append(' | ' + msg)

        msg = ''.join(L)
        sys.stdout.write(msg)
        for i in range(self.term_width-int(self.TOTAL_BAR_LENGTH)-len(msg)-3):
            sys.stdout.write(' ')

        # Go back to the center of the bar.
        for i in range(self.term_width-int(self.TOTAL_BAR_LENGTH/2)+2):
            sys.stdout.write('\b')
        sys.stdout.write(' %d/%d ' % (current+1, total))

        if current < total-1:
            sys.stdout.write('\r')
        else:
            sys.stdout.write('\n')
        sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def make_print_to_file(path='./'):
    '''
    path， it is a path for save your log about fuction print
    example:
    use  make_print_to_file()   and the   all the information of funtion print , will be write in to a log file
    :return:
    '''
    import sys
    import os
    import sys
    import datetime

    class Logger(object):
        def __init__(self, filename="Default.log", path="./"):
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), "a", encoding='utf8', )

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass

    fileName = datetime.datetime.now().strftime('day' + '%Y_%m_%d')
    sys.stdout = Logger(fileName + '.log', path=path)

    #############################################################
    # 这里输出之后的所有的输出的print 内容即将写入日志
    #############################################################
    print(fileName.center(60, '*'))

