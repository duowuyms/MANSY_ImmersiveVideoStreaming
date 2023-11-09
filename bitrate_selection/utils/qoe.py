from utils.common import normalize_quality


# to ensure that the all QoE metrics are in the same/similar scales
SCALE_QUALITY = 1
SCALE_VARIANCE = 1
SCALE_REBUFFER = 1


class QoEModel:
    def __init__(self, config, weight1, weight2, weight3):
        self.config = config
        self.weight1 = weight1
        self.weight2 = weight2
        self.weight3 = weight3
        self.qoe1 = 0.0
        self.qoe2 = 0.0
        self.qoe3 = 0.0
        self.prev_viewport_quality = None
        self.prev_rebuffer_time = 0.0

    def calculate_qoe(self, actual_viewport, tile_quality, rebuffer_time):
        viewport_quality = sum(actual_viewport * tile_quality) / sum(actual_viewport)
        intra_viewport_quality_variance = sum(actual_viewport * abs(tile_quality - viewport_quality)) / sum(actual_viewport)
        intra_viewport_quality_variance = normalize_quality(self.config, intra_viewport_quality_variance)
        viewport_quality = normalize_quality(self.config, viewport_quality)
        inter_viewport_quality_variance = abs(viewport_quality - self.prev_viewport_quality) if self.prev_viewport_quality is not None else 0.0
        self.prev_viewport_quality = viewport_quality
        self.prev_rebuffer_time = rebuffer_time
        self.qoe1 = viewport_quality * SCALE_QUALITY
        self.qoe2 = rebuffer_time * SCALE_REBUFFER
        self.qoe3 = (intra_viewport_quality_variance + inter_viewport_quality_variance) * SCALE_VARIANCE
        qoe = self.weight1 * self.qoe1 - self.weight2 * self.qoe2 - self.weight3 * self.qoe3
        return qoe, self.qoe1, self.qoe2, self.qoe3

    def reset(self):
        self.qoe1 = 0.0
        self.qoe2 = 0.0
        self.qoe3 = 0.0
        self.prev_viewport_quality = None
        self.prev_rebuffer_time = 0.0

    def reset_with_new_weights(self, weight1, weight2, weight3):
        self.weight1 = weight1
        self.weight2 = weight2
        self.weight3 = weight3
        self.reset()


class QoEModelExpert(QoEModel):
    def calculate_qoe_with_given_quality(self, viewport_quality, prev_viewport_quality, intra_viewport_quality_variance, rebuffer_time):
        viewport_quality = normalize_quality(self.config, viewport_quality)
        intra_viewport_quality_variance = normalize_quality(self.config, intra_viewport_quality_variance)
        inter_viewport_quality_variance = abs(viewport_quality - prev_viewport_quality) if prev_viewport_quality is not None else 0.0
        prev_viewport_quality = viewport_quality
        qoe1 = viewport_quality * SCALE_QUALITY
        qoe2 = rebuffer_time * SCALE_REBUFFER
        qoe3 = (intra_viewport_quality_variance + inter_viewport_quality_variance) * SCALE_VARIANCE
        qoe = self.weight1 * qoe1 - self.weight2 * qoe2 - self.weight3 * qoe3
        return qoe, qoe1, qoe2, qoe3, prev_viewport_quality
