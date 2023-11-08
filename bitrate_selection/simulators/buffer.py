
class PlaybackBuffer:
    def __init__(self, startup_download, chunk_length):
        # TODO: check whether such initialization is appropriate
        self.startup_download = startup_download
        self.chunk_length = chunk_length
        # self.buf_size = startup_download
        self.buf_size = chunk_length * 3

    def push_chunk(self, chunk_length, download_time):
        rebuf_time = 0.
        if download_time > self.buf_size:
            rebuf_time = download_time - self.buf_size
            self.buf_size = chunk_length
        else:
            self.buf_size = self.buf_size - download_time + chunk_length
        return rebuf_time

    def get_buffer_size(self):
        return self.buf_size

    def set_buffer_size(self, buf_size):
        self.buf_size = buf_size

    def reset(self):
        # self.buf_size = self.startup_download
        self.buf_size = self.chunk_length * 3
