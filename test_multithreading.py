# -*- coding=utf-8 -*-
#@author:liuAmon
#@contact:utopfish@163.com
#@file:test_multithreading.py
#@time: 2019/10/17 22:44
from multiprocessing import Process,Pool
from singleCharacterFitch import *
import threading
#多线程加速的测试
class RetThread(threading.Thread):

    def __init__(self, target=None, name=None, args=(), \
                 kwargs={}, createChild=True, daemon=None):
        threading.Thread.__init__(self, group=None, target=target, \
                                  name=name, args=args, kwargs=kwargs, daemon=daemon)
        self.__ret = None
        self.__stop = True
        self.__createChild = createChild
        self.__guard = None

    def run(self):
        self.__stop = True
        if (self.__createChild):
            guard = RetThread(target=self._target, args=self._args, \
                              kwargs=self._kwargs, createChild=False, daemon=True)
            self.__guard = guard
            self.__guard.start()
            while (self.__stop and self.__guard.is_alive()):
                pass
            del self._target, self._args, self._kwargs
        else:
            try:
                if self._target:
                    self.__ret = self._target(*self._args, **self._kwargs)
            finally:
                del self._target, self._args, self._kwargs

    def getReturn(self):
        if (self.__guard != None and self.__ret == None):
            self.__ret = self.__guard.getReturn()
        return self.__ret

    def stop(self):
        if not self._initialized:
            raise RuntimeError(self.getName() + " Thread.__init__() not called")
        if not self._started.is_set():
            raise RuntimeError("cannot stop thread until it is started")
        self.__stop = False




if __name__ == '__main__':
    path = r"testData\011号简化数据集奇虾\011号完整数据集.txt"
    data = readDataTxt(path)
    li = np.array(data)
    record=[]
    start=time.time()
    for i in range(1,len(li[0])):
        t = RetThread(target=getSingleChararcterFitch, args=(te[0],li[:,i],))
        record.append(t)
    for i in record:   # start threads 此处并不会执行线程，而是将任务分发到每个线程，同步线程。等同步完成后再开始执行start方法
        i.start()
        i.join()
    res=[]
    p=Pool(processes=4)
    for i in range(1,len(li[0])):
        r=p.apply(getSingleChararcterFitch,(te[0],li[:,i],))
        res.append(r)
    print(time.time()-start)

