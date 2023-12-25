import numpy as np
import os
from PIL import Image

NUM_CHANNELS = 3


# --------------------------------------------------------------------------- #

def img_read_helper(path):
    """
    Creates an RGBImage object from the given image file
    """
    # Open the image in RGB
    img = Image.open(path).convert("RGB")
    # Convert to numpy array and then to a list
    matrix = np.array(img).tolist()
    # Use student's code to create an RGBImage object
    return RGBImage(matrix)


def img_save_helper(path, image):
    """
    Saves the given RGBImage instance to the given path
    """
    # Convert list to numpy array
    img_array = np.array(image.get_pixels())
    # Convert numpy array to PIL Image object
    img = Image.fromarray(img_array.astype(np.uint8))
    # Save the image object to path
    img.save(path)


# --------------------------------------------------------------------------- #

# Part 1: RGB Image #
class RGBImage:
    """
    Represents an image in RGB format
    """

    def __init__(self, pixels):
        """
        Initializes a new RGBImage object

        # Test with non-rectangular list
        >>> pixels = [
        ...              [[255, 255, 255], [255, 255, 255]],
        ...              [[255, 255, 255]]
        ...          ]
        >>> RGBImage(pixels)
        Traceback (most recent call last):
        ...
        TypeError

        # Test instance variables
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img.pixels
        [[[255, 255, 255], [0, 0, 0]]]
        >>> img.num_rows
        1
        >>> img.num_cols
        2
        """
        # Raise exceptions here
        if not (
            isinstance(pixels, list) and
            pixels and
            all(isinstance(row, list) and row and len(row) == len(pixels[0]) for row in pixels) and
            all(isinstance(pixel, list) and len(pixel) == 3 for row in pixels for pixel in row) 
        ):
            raise TypeError#("Invalid pixels argument")
        if not(
            all(0 <= value <= 255 for value in (intensity for row in pixels for pixel in row for intensity in pixel))
        ):
            raise ValueError#("Please enter a value between 0 and 255, both inclusive")
        self.pixels = pixels
        self.num_rows = len(self.pixels)
        self.num_cols = len(self.pixels[0])

    def size(self):
        """
        Returns the size of the image in (rows, cols) format

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img.size()
        (1, 2)
        """
        return (self.num_rows, self.num_cols)

    def get_pixels(self):
        """
        Returns a copy of the image pixel array

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_pixels = img.get_pixels()

        # Check if this is a deep copy
        >>> img_pixels                               # Check the values
        [[[255, 255, 255], [0, 0, 0]]]
        >>> id(pixels) != id(img_pixels)             # Check outer list
        True
        >>> id(pixels[0]) != id(img_pixels[0])       # Check row
        True
        >>> id(pixels[0][0]) != id(img_pixels[0][0]) # Check pixel
        True
        """
        return [[pixel[:] for pixel in row] for row in self.pixels]
        #return [[pixel for pixel in row] for row in self.pixels]

    def copy(self):
        """
        Returns a copy of this RGBImage object

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_copy = img.copy()

        # Check that this is a new instance
        >>> id(img_copy) != id(img)
        True
        """
        new_pixels = self.get_pixels()
        return RGBImage(new_pixels)

    def get_pixel(self, row, col):
        """
        Returns the (R, G, B) value at the given position

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)

        # Test with an invalid index
        >>> img.get_pixel(1, 0)
        Traceback (most recent call last):
        ...
        ValueError

        # Run and check the returned value
        >>> img.get_pixel(0, 0)
        (255, 255, 255)
        """
        if not (isinstance(row, int) and isinstance(col, int)):
            raise TypeError("Row and col must be integers")
        if not (0 <= row < self.num_rows and 0 <= col < self.num_cols):
            raise ValueError#("Invalid index")
        return tuple(self.pixels[row][col])

    def set_pixel(self, row, col, new_color):
        """
        Sets the (R, G, B) value at the given position

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)

        # Test with an invalid new_color tuple
        >>> img.set_pixel(0, 0, (256, 0, 0))
        Traceback (most recent call last):
        ...
        ValueError

        # Check that the R/G/B value with negative is unchanged
        >>> img.set_pixel(0, 0, (-1, 0, 0))
        >>> img.pixels
        [[[255, 0, 0], [0, 0, 0]]]
        """
        if not (isinstance(row, int) and isinstance(col, int)):
            raise TypeError("Row and col must be integers")
        if not (0 <= row < self.num_rows and 0 <= col < self.num_cols):
            raise ValueError("Invalid row or col index")
        if not (
            isinstance(new_color, tuple) and
            len(new_color) == 3 and
            all(isinstance(value, int) for value in new_color)
        ):
            raise TypeError("Invalid new_color argument")
        if any(value > 255 for value in new_color):
            raise ValueError #("Intensity value in new_color cannot exceed 255")
        for i in range(3):
            if new_color[i] >= 0:
                self.pixels[row][col][i] = new_color[i] 


# Part 2: Image Processing Template Methods #
class ImageProcessingTemplate:
    """
    Contains assorted image processing methods
    Intended to be used as a parent class
    """

    def __init__(self):
        """
        Creates a new ImageProcessingTemplate object

        # Check that the cost was assigned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost
        0
        """
        self.cost = 0

    def get_cost(self):
        """
        Returns the current total incurred cost

        # Check that the cost value is returned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost = 50 # Manually modify cost
        >>> img_proc.get_cost()
        50
        """
        return self.cost

    def negate(self, image):
        """
        Returns a negated copy of the given image

        # Check if this is returning a new RGBImage instance
        >>> img_proc = ImageProcessingTemplate()
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_negate = img_proc.negate(img)
        >>> id(img) != id(img_negate) # Check for new RGBImage instance
        True

        # The following is a description of how this test works
        # 1 Create a processor
        # 2/3 Read in the input and expected output
        # 4 Modify the input
        # 5 Compare the modified and expected
        # 6 Write the output to file
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()                            # 1
        >>> img = img_read_helper('img/test_image_32x32.png')                 # 2
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_negate.png')  # 3
        >>> img_negate = img_proc.negate(img)                               # 4
        >>> img_negate.pixels == img_exp.pixels # Check negate output       # 5
        True
        >>> img_save_helper('img/out/test_image_32x32_negate.png', img_negate)# 6
        """
        if not isinstance(image, RGBImage):
            raise TypeError("The image must be an instance of RGBImage")

        negated_pixels = [[[255 - val for val in pixel] for pixel in row] for row in image.pixels]
        return RGBImage(negated_pixels)

    def grayscale(self, image):
        """
        Returns a grayscale copy of the given image

        # See negate for info on this test
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_gray.png')
        >>> img_gray = img_proc.grayscale(img)
        >>> img_gray.pixels == img_exp.pixels # Check grayscale output
        True
        >>> img_save_helper('img/out/test_image_32x32_gray.png', img_gray)
        """
        if not isinstance(image, RGBImage):
            raise TypeError#("The image must be an instance of RGBImage")

        try:
            grayscale_pixels = []
            for row in image.pixels:
                grayscale_row = []
                for pixel in row:
                    avg = sum(pixel) // 3
                    grayscale_row.append([avg, avg, avg])
                grayscale_pixels.append(grayscale_row)
            return RGBImage(grayscale_pixels)
        except Exception as e:
            raise TypeError(f"Error in creating grayscale image: {e}")

    def rotate_180(self, image):
        """
        Returns a rotated version of the given image

        # See negate for info on this test
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_rotate.png')
        >>> img_rotate = img_proc.rotate_180(img)
        >>> img_rotate.pixels == img_exp.pixels # Check rotate_180 output
        True
        >>> img_save_helper('img/out/test_image_32x32_rotate.png', img_rotate)
        """
        if not isinstance(image, RGBImage):
            raise TypeError("The image must be an instance of RGBImage")
        
        rotated_pixels = [list(reversed(row)) for row in reversed(image.get_pixels())]
        return RGBImage(rotated_pixels)

    def get_average_brightness(self, image):
        """
        Returns the average brightness for the given image

        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_proc.get_average_brightness(img)
        86
        """
        total_brightness= sum(sum(sum(pixel)//3 for pixel in row) for row in image.pixels)
        num_pixels= sum(len(row) for row in image.pixels)
        return total_brightness // num_pixels

    def adjust_brightness(self, image, intensity):
        """
        Returns a new image with adjusted brightness level

        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_adjusted.png')
        >>> img_adjust = img_proc.adjust_brightness(img, 75)
        >>> img_adjust.pixels == img_exp.pixels # Check adjust_brightness
        True
        >>> img_save_helper('img/out/test_image_32x32_adjusted.png', img_adjust)
        """
        if not isinstance(intensity, int):
            raise TypeError("Intensity must be an integer.")
        if intensity < -255 or intensity > 255:
            raise ValueError("Intensity must be between -255 and 255.")
        adjusted_pixels = [[[max(0, min(255, val + intensity)) for val in pixel] for pixel in row] for row in image.pixels]
        return RGBImage(adjusted_pixels) 

    def blur(self, image):
        """
        Returns a new image with the pixels blurred

        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_blur.png')
        >>> img_blur = img_proc.blur(img)
        >>> img_blur.pixels == img_exp.pixels # Check blur
        True
        >>> img_save_helper('img/out/test_image_32x32_blur.png', img_blur)
        """
        def get_average_of_neighbors(pixels, x, y):
            neighbors = [
                pixels[i][j]
                for i in range(max(0, x-1), min(len(pixels), x+2))
                for j in range(max(0, y-1), min(len(pixels[0]), y+2))
            ]
            average = [sum(values) // len(neighbors) for values in zip(*neighbors)]
            return average

        blurred_pixels = [
            [get_average_of_neighbors(image.pixels, row_i, col_i) for col_i in range(len(row))]
            for row_i, row in enumerate(image.pixels)
        ]
        return RGBImage(blurred_pixels)


# Part 3: Standard Image Processing Methods #
class StandardImageProcessing(ImageProcessingTemplate):
    """
    Represents a standard tier of an image processor
    """

    def __init__(self):
        """
        Creates a new StandardImageProcessing object

        # Check that the cost was assigned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost
        0
        """
        # YOUR CODE GOES HERE #
        super().__init__()
        self.cost = 0
        self.free_operations = 0

    def negate(self, image):
        """
        Returns a negated copy of the given image

        # Check the expected cost
        >>> img_proc = StandardImageProcessing()
        >>> img_in = img_read_helper('img/square_32x32.png')
        >>> negated = img_proc.negate(img_in)
        >>> img_proc.get_cost()
        5

        # Check that negate works the same as in the parent class
        >>> img_proc = StandardImageProcessing()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_negate.png')
        >>> img_negate = img_proc.negate(img)
        >>> img_negate.pixels == img_exp.pixels # Check negate output
        True
        """
        negated_image = super().negate(image)
        if self.free_operations > 0:
            self.free_operations -= 1
        else:
            self.cost += 5
        return negated_image

    def grayscale(self, image):
        """
        Returns a grayscale copy of the given image

        """
        grayscale_image = super().grayscale(image)
        if self.free_operations > 0:
            self.free_operations -= 1
        else:
            self.cost += 6
        return grayscale_image

    def rotate_180(self, image):
        """
        Returns a rotated version of the given image
        """
        rotated_image = super().rotate_180(image)
        if self.free_operations > 0:
            self.free_operations -= 1
        else:
            self.cost += 10
        return rotated_image

    def adjust_brightness(self, image, intensity):
        """
        Returns a new image with adjusted brightness level
        """
        adjusted_image = super().adjust_brightness(image, intensity)
        if self.free_operations > 0:
            self.free_operations -= 1
        else:
            self.cost += 1
        return adjusted_image

    def blur(self, image):
        """
        Returns a new image with the pixels blurred
        """
        blurred_image = super().blur(image)
        if self.free_operations > 0:
            self.free_operations -= 1
        else:
            self.cost += 5
        return blurred_image

    def redeem_coupon(self, amount):
        """
        Makes the given number of methods calls free

        # Check that the cost does not change for a call to negate
        # when a coupon is redeemed
        >>> img_proc = StandardImageProcessing()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_proc.redeem_coupon(1)
        >>> img = img_proc.rotate_180(img)
        >>> img_proc.get_cost()
        0
        """
        if not isinstance(amount, int):
            raise TypeError("Amount must be an integer.")
        if amount <= 0:
            raise ValueError("Amount must be a positive integer.")
        self.free_operations += amount


# Part 4: Premium Image Processing Methods #
class PremiumImageProcessing(ImageProcessingTemplate):
    """
    Represents a paid tier of an image processor
    """

    def __init__(self):
        """
        Creates a new PremiumImageProcessing object

        # Check the expected cost
        >>> img_proc = PremiumImageProcessing()
        >>> img_proc.get_cost()
        50
        """
        # YOUR CODE GOES HERE #
        super().__init__()
        self.cost = 50

    def chroma_key(self, chroma_image, background_image, color):
        """
        Returns a copy of the chroma image where all pixels with the given
        color are replaced with the background image.

        # Check output
        >>> img_proc = PremiumImageProcessing()
        >>> img_in = img_read_helper('img/square_32x32.png')
        >>> img_in_back = img_read_helper('img/test_image_32x32.png')
        >>> color = (255, 255, 255)
        >>> img_exp = img_read_helper('img/exp/square_32x32_chroma.png')
        >>> img_chroma = img_proc.chroma_key(img_in, img_in_back, color)
        >>> img_chroma.pixels == img_exp.pixels # Check chroma_key output
        True
        >>> img_save_helper('img/out/square_32x32_chroma.png', img_chroma)
        """
        if not all(isinstance(image, RGBImage) for image in (chroma_image, background_image)):
            raise TypeError("The images must be instances of RGBImage")
        if chroma_image.size() != background_image.size():
            raise ValueError("The images must have the same dimensions")
        
        for i in range(chroma_image.size()[0]):
            for j in range(chroma_image.size()[1]):
                if chroma_image.get_pixel(i, j) == color:
                    chroma_image.set_pixel(i, j, background_image.get_pixel(i, j))
                    
        return chroma_image

    def sticker(self, sticker_image, background_image, x_pos, y_pos):
        """
        Returns a copy of the background image where the sticker image is
        placed at the given x and y position.

        # Test with out-of-bounds image and position size
        >>> img_proc = PremiumImageProcessing()
        >>> img_sticker = img_read_helper('img/square_6x6.png')
        >>> img_back = img_read_helper('img/test_image_32x32.png')
        >>> x, y = (31, 0)
        >>> img_proc.sticker(img_sticker, img_back, x, y)
        Traceback (most recent call last):
        ...
        ValueError

        # Check output
        >>> img_proc = PremiumImageProcessing()
        >>> img_sticker = img_read_helper('img/square_6x6.png')
        >>> img_back = img_read_helper('img/test_image_32x32.png')
        >>> x, y = (3, 3)
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_sticker.png')
        >>> img_combined = img_proc.sticker(img_sticker, img_back, x, y)
        >>> img_combined.pixels == img_exp.pixels # Check sticker output
        True
        >>> img_save_helper('img/out/test_image_32x32_sticker.png', img_combined)
        """
        if not all(isinstance(image, RGBImage) for image in (sticker_image, background_image)):
            raise TypeError#("The images must be instances of RGBImage")
        if not isinstance(x_pos, int) or not isinstance(y_pos, int):
            raise TypeError#("The position must be an integer")
        if sticker_image.size()[0] + x_pos > background_image.size()[0] or sticker_image.size()[1] + y_pos > background_image.size()[1]:
            raise ValueError#("The sticker image must fit within the background image")
        
        for i in range(sticker_image.size()[0]):
            for j in range(sticker_image.size()[1]):
                background_image.set_pixel(i + x_pos, j + y_pos, sticker_image.get_pixel(i, j))
        return background_image
        

    def edge_highlight(self, image):
        """
        Returns a new image with the edges highlighted

        # Check output
        >>> img_proc = PremiumImageProcessing()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_edge = img_proc.edge_highlight(img)
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_edge.png')
        >>> img_exp.pixels == img_edge.pixels # Check edge_highlight output
        True
        >>> img_save_helper('img/out/test_image_32x32_edge.png', img_edge)
        """
        if not isinstance(image, RGBImage):
            raise TypeError("The image must be an instance of RGBImage")
        
        grayscale_pixels = [[sum(pixel) // 3 for pixel in row] for row in image.get_pixels()]

        kernel = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
        result = image.copy()
        for i in range(image.num_rows):
            for j in range(image.num_cols):
                val = 0
                for ki in range(-1, 2):
                    for kj in range(-1, 2):
                        ni, nj = i + ki, j + kj
                        if 0 <= ni < image.num_rows and 0 <= nj < image.num_cols:
                            pixel_val = sum(image.get_pixel(ni, nj)) // 3
                            val += pixel_val * kernel[ki + 1][kj + 1]
                val = max(0, min(255, val))
                result.set_pixel(i, j, (val, val, val))
        return result


# Part 5: Image KNN Classifier #
class ImageKNNClassifier:
    """
    Represents a simple KNNClassifier
    """

    def __init__(self, k_neighbors):
        """
        Creates a new KNN classifier object
        """
        self.k_neighbors = k_neighbors

    def fit(self, data):
        """
        Stores the given set of data and labels for later
        """
        if len(data) < self.k_neighbors:
            raise ValueError("Not enough data to fit the classifier")
        self.data = data

    def distance(self, image1, image2):
        """
        Returns the distance between the given images

        >>> img1 = img_read_helper('img/steve.png')
        >>> img2 = img_read_helper('img/knn_test_img.png')
        >>> knn = ImageKNNClassifier(3)
        >>> knn.distance(img1, img2)
        15946.312896716909
        """
        if not all(isinstance(image, RGBImage) for image in (image1, image2)):
            raise TypeError("The images must be instances of RGBImage")
        if image1.size() != image2.size():
            raise ValueError("The images must have the same dimensions")
        
        diff_squared= sum(
            (val1- val2) **2
            for row1, row2 in zip(image1.get_pixels(), image2.get_pixels())
            for pixel1, pixel2 in zip(row1, row2)
            for val1, val2 in zip(pixel1, pixel2)
        )
        return diff_squared ** 0.5
        

    def vote(self, candidates):
        """
        Returns the most frequent label in the given list

        >>> knn = ImageKNNClassifier(3)
        >>> knn.vote(['label1', 'label2', 'label2', 'label2', 'label1'])
        'label2'
        """
        label_counts = {}
        for label in candidates:
            label_counts[label] = label_counts.get(label, 0) + 1

        most_common_label = None
        max_count = 0

        for label, count in label_counts.items():
            if count > max_count:
                most_common_label = label
                max_count = count

        return most_common_label

    def predict(self, image):
        """
        Predicts the label of the given image using the labels of
        the K closest neighbors to this image

        The test for this method is located in the knn_tests method below
        """
        if not self.data:
            raise ValueError()
        distances = [(self.distance(image, data[0]), data[1]) for data in self.data]
        distances.sort(key=lambda x: x[0])
        neighbors = [label for _, label in distances[:self.k_neighbors]]
        return self.vote(neighbors)

def knn_tests(test_img_path):
    """
    Function to run knn tests

    >>> knn_tests('img/knn_test_img.png')
    'nighttime'
    """
    # Read all of the sub-folder names in the knn_data folder
    # These will be treated as labels
    path = 'knn_data'
    data = []
    for label in os.listdir(path):
        label_path = os.path.join(path, label)
        # Ignore non-folder items
        if not os.path.isdir(label_path):
            continue
        # Read in each image in the sub-folder
        for img_file in os.listdir(label_path):
            train_img_path = os.path.join(label_path, img_file)
            img = img_read_helper(train_img_path)
            # Add the image object and the label to the dataset
            data.append((img, label))

    # Create a KNN-classifier using the dataset
    knn = ImageKNNClassifier(5)

    # Train the classifier by providing the dataset
    knn.fit(data)

    # Create an RGBImage object of the tested image
    test_img = img_read_helper(test_img_path)

    # Return the KNN's prediction
    predicted_label = knn.predict(test_img)
    return predicted_label
