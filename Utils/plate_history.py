from collections import defaultdict, deque
from difflib import SequenceMatcher


class PlateHistory:

    def __init__(self):

        # lebih panjang biar stabil
        self.plate_history = defaultdict(lambda: deque(maxlen=25))

        self.plate_final = {}


    def get_box(self, x1, x2, y1, y2):

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        return f"{cx//20}_{cy//20}"

    def _similarity(self, a, b):
        return SequenceMatcher(None, a, b).ratio()


    def get_stable_plate(self, box_id, new_text):

        if not new_text:
            return self.plate_final.get(box_id, "")

        history = self.plate_history[box_id]

        history.append(new_text)

        
        scores = {}

        for text in history:

            if text not in scores:
                scores[text] = 0

            
            for other in history:
                if text == other:
                    scores[text] += 1
                else:
                    scores[text] += self._similarity(text, other)

        best_text = max(scores.items(), key=lambda x: x[1])[0]

        self.plate_final[box_id] = best_text

        return best_text