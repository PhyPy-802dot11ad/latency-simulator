import simpy

from .Log import Log


class Buffer():
    """Class representing a (input) buffer.

    :param GET_REQUESTED: Identifier of the 'get requested' event.
    :type GET_REQUESTED: int
    :param PUT: Identifier of the 'put' event.
    :type PUT: int
    :param GET_GRANTED: Identifier of the 'get granted' event.
    :type GET_GRANTED: int
    :param env: Simpy environment.
    :type env: object
    :param store: Simpy store.
    :type store: object
    :param log: Event log.
    :type log: object
    """

    GET_REQUESTED = 0
    PUT = 1
    GET_GRANTED = 2

    def __init__(self, env, log_name, log_path, log_subdir):
        """
        :param env: Simpy environment.
        :type env: object
        """

        self.env = env
        self.store = simpy.Store(env)
        # self.log = Log((0,3))
        self.log = Log(log_name, ['Count', 'Event'], path=log_path, subdir_name=log_subdir)

    def put(self, item):
        """Put item into the buffer.

        Stores the item in the Simpy store and save the event to the log.

        :param item: Input item.
        :type item: any
        """

        self.store.put(item)
        self.log_put()

    def get(self):
        """Asks for the first item from the buffer.

        Ask for the first item from the Simpy store and log the event.

        :return: Simpy.store.get() handler
        :rtype: object
        """
        self.log_get_request()
        return self.store.get()

    def get_queue_size(self):
        """Gets the amount of items in the buffer.

        :return: Number of items in buffer.
        :rtype: int
        """
        return len(self.store.items)

    def log_get_request(self):
        """Logs a get request event in the local log memory.
        """
        self.log.append( self.env.now, self.get_queue_size(), self.GET_REQUESTED )

    def log_put(self):
        """Logs a put event in the local log memory.
        """
        self.log.append( self.env.now, self.get_queue_size(), self.PUT )

    def log_get_granted(self):
        """Logs a get granted event in the local log memory.
        """
        self.log.append( self.env.now, self.get_queue_size(), self.GET_GRANTED )

    def save_log(self, subdir, name):
        """Save the log to a file.

        :param subdir: Log subdirectory path.
        :type subdir: str
        :param name: Filename.
        :type name: str
        """
        self.log.save(subdir, name)