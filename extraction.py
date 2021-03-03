import sys

import pytesseract
import tensorflow as tf

from tablenet import image_util

def load_model():
    model = tf.keras.models.load_model('saved_model')
    return model

def print_usage():
    pass

def apply_mask(image, table_mask, column_mask):
    pass

class BoundingBox:
    def __init__(self, top, left, width, height):
        self.top = top
        self.left = left
        self.width = width
        self.height = height

    def as_tuple():
        return (self.top, self.left, self.width, self.height)

class TableExtractor:
    CONFIDENCE_THRESHOLD = 60

    def __init__(self):
        # The average height of text, useful for grouping row together
        self._avg_text_height = None
        # List of tops to mark text on a diffent row
        self._tops = [] 
        # List of column ranges (left, right) to mark different columns
        self._column_ranges = []
        # Store OCR output
        self._ocr_results = []

    def extract_table(self, image, table_mask, column_mask):
        self._reset_state()

        filtered_image = apply_mask(image, table_mask, column_mask)
        self._extract_words(filtered_image)
        self._calculate_avg_text_height()
        self._populate_tops()
        self._populate_column_ranges()

        rows = len(self._tops)
        cols = len(self._column_ranges)
        table = [[""] * cols] * rows
        for res in self._ocr_results:
            tup = self._cell_index(res.bounding_box)
            if tup is not None:
                (row, col) = tup
                table[row][col] = res.text

        return table

    def _reset_state(self):
        self._avg_text_height = None
        self._tops = [] 
        self._column_ranges = []
        self._ocr_results = []

    def _extract_words(self, image):
        d = pytesseract.image_to_data(filtered_image, output_type=Output.DICT)
        n_boxes = len(d['text'])
        for i in range(n_boxes):
            # Only find text within the confidence threshold
            if int(d['conf'][i] > CONFIDENCE_THRESHOLD:
                (t, l, w, h) = (d['top'][i], d['left'][i], d['width'][i], d['height'][i])
                text = d['text'][i]
                self._ocr_results.push(OcrResult(t, l, w, h, text))

    def _calculate_avg_text_height(self):
        height_total = 0.0
        height_count = 0
        for res in self._ocr_results:
            (_, _, _, h) = res.bounding_box.as_tuple()
            height_total = height_total + h
            height_count = height_count + 1
        self._avg_text_height = height_total / height_count

    def _populate_tops(self):
        """Group text by top, this in effect group text by row"""
        for res in self._ocr_results:
            (t, _, _, _) = res.bounding_box.as_tuple()
            idx = 0
            found = False
            while idx < len(self._tops):
                top = self._tops[i]
                lo = top - self._avg_text_height / 2
                hi = top + self._avg_text_height / 2
                if t < lo:
                    break
                elif lo <= t and t <= hi:
                    found = True
                    break
                idx += 1
            if not found:
                self._tops.insert(idx, t)

    def _populate_column_ranges(self):
        for res in self._ocr_results:
            (_, left, width, _) = res.bounding_box.as_tuple()
            right = left + width
            idx = 0
            found = False
            while idx < len(self._column_ranges):
                rng = self._column_ranges[idx]
                if right < range[0]:
                    break
                elif is_intersect((left, right), rng):
                    found = True
                    break
                idx += 1
            if not found:
                self._column_ranges.insert(idx, (left, right))
            else:
                self._column_ranges[idx] = union(self._column_ranges[idx], (left, right))

    def _cell_index(self, bounding_box):
        """
        Return the row and column index for a given bounding box.

        It returns None if the row and column cannot be determined.
        """
        t, l, w, h = bounding_box.as_tuple()
        row = -1
        col = -1
        for i in len(self.tops):
            top = tops[i]
            if top - avg_height / 2 <= t and t <= top + avg_height / 2:
                row = i
                break
        for i in len(self.column_dims):
            if is_intersect((l, l+w), column_dims[i]):
                col = i
                break
        if row == -1 or col == -1:
            return None
        else:
            return (row, col)

    class OcrResult:
        def __init__(self, top, left, width, height, text):
            self.bounding_box = BoundingBox(top, left, width, height)
            self.text = text

def is_intersect(range1, range2):
    """Find out if two ranges intersect"""
    l1, r1 = range1
    l2, r2 = range2
    return (l2 <= l1 and l1 <= r2) or (l2 <= r1 and r1 <= r2)

def union(range1, range2):
    """Return the union of two ranges"""
    l1, r1 = range1
    l2, r2 = range2
    return (min(l1, l2), max(r1, r2))

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print_usage()
        sys.exit(0)

    file_name = sys.argv[1]
    image_file = tf.io.read_file(file_path)
    images = tf.reshape(image_util.decode_jpeg(image_file), (1, 256, 256, 3))

    model = load_model()
    table_masks, column_masks = model.predict(image)

