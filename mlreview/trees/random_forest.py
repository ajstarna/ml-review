'''
https://link.springer.com/content/pdf/10.1023%2FA%3A1010933404324.pdf
'''


class RandomForest:

    def __init__(self, max_depth=None):
        if max_depth is None:
            max_depth = float('inf')
        self.max_depth = max_depth

        
