import re


class PlateRegex:

    def __init__(self):

        self.plate_pattern = re.compile(
            r'^[A-Z]{1,2}[0-9]{1,4}[A-Z]{1,3}$'
        )

    def validate(
        self,
        text
    ):

        return self.plate_pattern.match(
            text
        )