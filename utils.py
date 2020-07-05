import os
import time
from threading import Thread
from queue import Queue
from datetime import datetime as dt


def print_timed(message, indentation=0, start='', end='\n'):
    now = dt.now().isoformat(sep=" ", timespec="milliseconds")
    print(start + now, '│\t' * indentation + f"{message: <128}", end=end)


class ProgressBar(object):
    class PrintAsync(Thread):
        def __init__(self, queue: Queue, interval=1.0):
            Thread.__init__(self)
            self._queue = queue
            self._interval = interval

        def run(self):
            cls = type(self)
            data = []

            while True:
                last_update = time.clock()

                if not self._queue.empty():
                    while not self._queue.empty():
                        data = self._queue.get(False)
                else:
                    data = self._queue.get(True)

                if data is None:
                    break

                cls._print(data)
                passed = time.clock() - last_update

                if passed < self._interval:
                    time.sleep(self._interval - passed)

        @staticmethod
        def _print(data):
            printable = []
            title_spacing = 0
            prefix_spacing = 0
            value_spacing = 0
            maximum_spacing = 0
            length_spacing = 0

            for record in data:
                title_spacing = max(title_spacing, len(record['title']))
                prefix_spacing = max(prefix_spacing, len(record['prefix']))
                value_spacing = max(value_spacing, len(str(record['value'])))
                maximum_spacing = max(maximum_spacing, len(str(record['maximum'])))
                length_spacing = max(length_spacing, record['length'])

            for record in data:
                filled_length = int(record['length'] * record['value'] // record['maximum'])
                bar = '│' + '█' * filled_length + '-' * (record['length'] - filled_length) + '│'
                printable.append(f'{record["update_time"].isoformat(sep=" ", timespec="milliseconds")}   '
                                 f'{record["title"]: >{title_spacing}}: '
                                 f'{record["prefix"]: <{prefix_spacing}} '
                                 f'{bar: <{length_spacing + 2}} '
                                 f'{record["value"]: >{value_spacing}} / '
                                 f'{record["maximum"]: <{maximum_spacing}} '
                                 f'{record["suffix"]}')

            os.system('cls')
            print("\n".join(printable))

    _progress_bars = set()
    _hash_max = 0
    _callback_queue = Queue()
    _update_thread = None

    def __new__(cls, *args, **kwargs):
        self = super().__new__(cls)
        self._hash = cls._hash_max
        cls._hash_max += 1
        cls._progress_bars.add(self)
        return self

    def __init__(self, value, maximum, title='', prefix='', suffix='', length=50):
        self._value = value
        self._maximum = maximum
        self._title = title
        self._prefix = prefix
        self._suffix = suffix
        self._length = length
        self._redraw()

    def __reduce__(self):
        return self.__class__, (self._value, self._maximum, self._title, self._prefix, self._suffix, self._length)

    def __hash__(self):
        return self._hash

    def __del__(self):
        self.dispose()

    @classmethod
    def init_renderer(cls, interval=1.0):
        if cls._update_thread is None:
            cls._update_thread = cls.PrintAsync(cls._callback_queue, interval)
            cls._update_thread.start()

    @classmethod
    def terminate_renderer(cls):
        cls._callback_queue.put(None, block=True)
        cls._update_thread = None

    def update(self, value=None, maximum=None, title=None, prefix=None, suffix=None, length=None):
        if value is not None:
            self._value = value

        if maximum is not None:
            self._maximum = maximum

        if title is not None:
            self._title = title

        if prefix is not None:
            self._prefix = prefix

        if suffix is not None:
            self._suffix = suffix

        if length is not None:
            self._length = length

        self._redraw()

    @classmethod
    def _redraw_all(cls):
        printable = [cls._data(progress_bar) for progress_bar in cls._progress_bars]
        cls._callback_queue.put(printable, block=False)

    def _redraw(self):
        self._update_time = dt.now()
        type(self)._redraw_all()

    def _data(self):
        return {
            'value': self._value,
            'maximum': self._maximum,
            'title': self._title,
            'prefix': self._prefix,
            'suffix': self._suffix,
            'length': self._length,
            'update_time': self._update_time
        }

    def __str__(self):
        filled_length = int(self._length * self._value // self._maximum)
        bar = '│' + '█' * filled_length + '-' * (self._length - filled_length) + '│'
        return f'{self._update_time.isoformat(sep=" ", timespec="milliseconds")}   ' \
               f'{self._title}: ' \
               f'{self._prefix} ' \
               f'{bar} ' \
               f'{self._value} / ' \
               f'{self._maximum} ' \
               f'{self._suffix}'

    def dispose(self):
        cls = type(self)
        cls._progress_bars.remove(self)
        cls._redraw_all()

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value
        self._redraw()

    @property
    def maximum(self):
        return self._maximum

    @maximum.setter
    def maximum(self, value):
        cls = type(self)
        self._maximum = value
        cls._portion_spacing = max(cls._portion_spacing, len(str(value)))
        self._redraw()

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, value):
        cls = type(self)
        self._title = value
        cls._title_spacing = max(cls._title_spacing, len(value))
        self._redraw()

    @property
    def prefix(self):
        return self._prefix

    @prefix.setter
    def prefix(self, value):
        cls = type(self)
        self._prefix = value
        cls._prefix_spacing = max(cls._prefix_spacing, len(value))
        self._redraw()

    @property
    def suffix(self):
        return self._suffix

    @suffix.setter
    def suffix(self, value):
        self._suffix = value
        self._redraw()

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, value):
        self._length = value
        self._redraw()
