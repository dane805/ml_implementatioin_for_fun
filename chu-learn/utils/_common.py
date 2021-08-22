import time

class Common:

    def __init__(self):
        pass

    def _print_end_message(self, comment):
        """Print verbose message on end."""
        if self.verbose >= 1:
            print(comment)
