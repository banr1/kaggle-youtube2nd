import numpy as np
import heapq
import random
import numbers

class AveragePrecisionCalculator(object):
    def __init__(self, top_n=None):
        if not ((isinstance(top_n, int) and top_n >= 0) or top_n is None):
            raise ValueError("top_n must be a positive integer or None.")
        self._top_n = top_n
        self._total_positives = 0
        self._heap = []

    @property
    def heap_size(self):
        return len(self._heap)

    @property
    def num_accumulated_positives(self):
        return self._total_positives

    def accumulate(self, predictions, actuals, num_positives=None):
        if len(predictions) != len(actuals):
            raise ValueError("the shape of predictions "\
                             "and actuals does not match.")
        if not num_positives is None:
            if not isinstance(num_positives, numbers.Number) or num_positives<0:
                raise ValueError("'num_positives' was provided "\
                                 "but it wan't a nonzero number.")
        if not num_positives is None:
            self._total_positives += num_positives
        else:
            self._total_positives += np.size(np.where(actuals > 0))
        topk = self._top_n
        heap = self._heap
        for i in range(np.size(predictions)):
            if topk is None or len(heap) < topk:
                heapq.heappush(heap, (predictions[i], actuals[i]))
            else:
                if predictions[i] > heap[0][0]:
                    heapq.heappop(heap)
                    heapq.heappush(heap, (predictions[i], actuals[i]))

    def clear(self):
        self._heap = []
        self._total_positives = 0

    def peek_ap_at_n(self):
        if self.heap_size <= 0:
            return 0
        predlists = np.array(list(zip(*self._heap)))
        ap = self.ap_at_n(predlists[0],
                          predlists[1],
                          n=self._top_n,
                          total_num_positives=self._total_positives)
        return ap

    @staticmethod
    def ap(predictions, actuals):
        return AveragePrecisionCalculator.ap_at_n(predictions,
                                                  actuals,
                                                  n=None)

    @staticmethod
    def ap_at_n(predictions, actuals, n=20, total_num_positives=None):
        if len(predictions) != len(actuals):
            raise ValueError("the shape of predictions "\
                             "and actuals does not match.")
        if n is not None:
            if not isinstance(n, int) or n <= 0:
                raise ValueError("n must be 'None' or a positive integer. "\
                                 f"It was '{n}'.")
        ap = 0.0
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        predictions, actuals = AveragePrecisionCalculator._shuffle(predictions,
                                                                   actuals)
        sortidx = sorted(range(len(predictions)),
                         key=lambda k: predictions[k],
                         reverse=True)
        if total_num_positives is None:
            numpos = np.size(np.where(actuals > 0))
        else:
            numpos = total_num_positives
        if numpos == 0:
            return 0
        if n is not None:
            numpos = min(numpos, n)
        delta_recall = 1.0 / numpos
        poscount = 0.0
        r = len(sortidx)
        if n is not None:
            r = min(r, n)
        for i in range(r):
            if actuals[sortidx[i]] > 0:
                poscount += 1
                ap += poscount / (i + 1) * delta_recall
        return ap

    @staticmethod
    def _shuffle(predictions, actuals):
        random.seed(0)
        suffidx = random.sample(range(len(predictions)), len(predictions))
        predictions = predictions[suffidx]
        actuals = actuals[suffidx]
        return predictions, actuals

    @staticmethod
    def _zero_one_normalize(predictions, epsilon=1e-7):
        denominator = np.max(predictions) - np.min(predictions)
        ret = (predictions - np.min(predictions)) / np.max(denominator, epsilon)
        return ret

class MeanAveragePrecisionCalculator(object):
    def __init__(self, num_class):
        if not isinstance(num_class, int) or num_class <= 1:
            raise ValueError("num_class must be a positive integer.")
        self._ap_calculators = []
        self._num_class = num_class
        for i in range(num_class):
            self._ap_calculators.append(
                AveragePrecisionCalculator())

    def accumulate(self, predictions, actuals, num_positives=None):
        if not num_positives:
            num_positives = [None for i in predictions.shape[1]]
        calculators = self._ap_calculators
        for i in range(len(predictions)):
            calculators[i].accumulate(predictions[i], actuals[i],
                                      num_positives[i])

    def clear(self):
        for calculator in self._ap_calculators:
            calculator.clear()

    def is_empty(self):
        return ([calculator.heap_size for calculator in self._ap_calculators] \
                == [0 for _ in range(self._num_class)])

    def peek_map_at_n(self):
        aps = [self._ap_calculators[i].peek_ap_at_n()
               for i in range(self._num_class)]
        return aps
