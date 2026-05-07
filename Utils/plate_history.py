from collections import defaultdict, deque


class PlateHistory:

    def __init__(self):

        self.plate_history = defaultdict(
            lambda: deque(maxlen=10)
        )

        self.plate_final = {}

    def get_box(
        self,
        x1,
        x2,
        y1,
        y2
    ):

        return f"{int(x1//10)}_{int(x2//10)}_{int(y1//10)}_{int(y2//10)}"

    def get_stable_plate(
        self,
        box_id,
        new_text
    ):

        if new_text:

            self.plate_history[
                box_id
            ].append(new_text)

            most_common = max(
                set(self.plate_history[box_id]),
                key=self.plate_history[box_id].count
            )

            self.plate_final[
                box_id
            ] = most_common

        return self.plate_final.get(
            box_id,
            ""
        )