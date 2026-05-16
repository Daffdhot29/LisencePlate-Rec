import re


class PlateRegex:

    def __init__(self):

      
        self.plate_pattern = re.compile(
            r'^[A-Z]{1,2}[0-9]{1,4}[A-Z]{1,3}$'
        )


    def _clean(self, text):

        if not text:
            return ""

        text = text.upper()
        text = text.replace(" ", "")
        text = re.sub(r'[^A-Z0-9]', '', text)

        return text

   
    def validate(self, text):

        text = self._clean(text)

        return bool(self.plate_pattern.match(text))

 
    def normalize(self, text):

        text = self._clean(text)

        if self.plate_pattern.match(text):
            return text

        return ""