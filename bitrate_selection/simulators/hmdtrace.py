import pickle


class HMDTrace:
    def __init__(self, viewport_path, tile_num_width, tile_num_height):
        self.viewports = pickle.load(open(viewport_path, 'rb'))
        self.tile_num_width = tile_num_width
        self.tile_num_height = tile_num_height
        self.start_chunk = self.viewports[0][0]
        self.end_chunk = self.viewports[-1][0]
        self.chunk_num = self.end_chunk - self.start_chunk + 1

    def get_hmd_trace_info(self):
        return self.start_chunk, self.end_chunk, self.chunk_num

    def get_viewport(self, chunk, flatten=True):
        gt_viewport  = self.viewports[chunk - self.start_chunk][1]
        pred_viewport = self.viewports[chunk - self.start_chunk][2]
        accuracy = self.viewports[chunk - self.start_chunk][3]
        if not flatten:
            gt_viewport = gt_viewport.reshape(self.tile_num_height, self.tile_num_width)
            pred_viewport = pred_viewport.reshape(self.tile_num_height, self.tile_num_width)
        return gt_viewport, pred_viewport, accuracy
