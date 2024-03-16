import cv2
import numpy as np

def filter_blur(image):
    """
    Applies a blur filter to the input image using a 5x5 kernel.

    Parameters:
    image (numpy.ndarray): The input image.

    Returns:
    numpy.ndarray: The blurred image.
    """
    
    imagem_media = cv2.blur(image, (5, 5))  # Kernel 5x5
    cv2.imwrite(f'./preprocessed/imagem_media.tif', imagem_media)
    
    return imagem_media

def filter_median(image):
    """
    Applies a median filter to the input image using a 5x5 kernel.

    Parameters:
    image (numpy.ndarray): The input image.

    Returns:
    numpy.ndarray: The image after median filtering.
    """
    imagem_mediana = cv2.medianBlur(image, 5)  # Kernel 5x5
    cv2.imwrite(f'./preprocessed/imagem_mediana.tif', imagem_mediana)
    
    return imagem_mediana

def filter_gaussian(image):
    """
    Applies a Gaussian blur filter to the input image using a 5x5 kernel.

    Parameters:
    image (numpy.ndarray): The input image.

    Returns:
    numpy.ndarray: The image after Gaussian blurring.
    """
    imagem_gaussiana = cv2.GaussianBlur(image, (5, 5), 0)  # Kernel 5x5
    cv2.imwrite(f'./preprocessed/imagem_gaussiana.tif', imagem_gaussiana)

    return imagem_gaussiana

def filter_sobel(image):
    """
    Applies the Sobel operator to detect edges in the input image.

    Parameters:
    image (numpy.ndarray): The input image.

    Returns:
    numpy.ndarray: The image containing edges detected by the Sobel operator.
    """
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    sobel_x = cv2.convertScaleAbs(sobel_x)
    sobel_y = cv2.convertScaleAbs(sobel_y)
    sobel = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

    cv2.imwrite(f'./preprocessed/sobel_edges.tif', sobel)
    
    return sobel

def filter_laplace(image):
    """
    Applies the Laplace operator to detect edges in the input image.

    Parameters:
    image (numpy.ndarray): The input image.

    Returns:
    numpy.ndarray: The image containing edges detected by the Laplace operator.
    """
    laplacian = cv2.Laplacian(image, cv2.CV_64F)

    # Converter para uint8
    laplacian = cv2.convertScaleAbs(laplacian)

    # Salvar a imagem resultante
    cv2.imwrite(f'./preprocessed/laplacian_edges.tif', laplacian)
    
    return laplacian

def erosion(image):
    """
    Applies erosion to the input image.

    Parameters:
    image (numpy.ndarray): The input image.

    Returns:
    numpy.ndarray: The image after erosion.
    """
    kernel = np.ones((5, 5), np.uint8)

    erosion = cv2.erode(image, kernel, iterations=1)

    cv2.imwrite(f'./preprocessed/erosion_result.tif', erosion)
    
    return erosion


def dilation(image):
    """
    Applies dilation to the input image.

    Parameters:
    image (numpy.ndarray): The input image.

    Returns:
    numpy.ndarray: The image after dilation.
    """
    kernel = np.ones((5, 5), np.uint8)
    
    dilation = cv2.dilate(image, kernel, iterations=1)
    
    cv2.imwrite(f'./preprocessed/dilation_result.tif', dilation)
    
    return dilation


def opening(image):
    """
    Applies opening to the input image.

    Parameters:
    image (numpy.ndarray): The input image.

    Returns:
    numpy.ndarray: The image after opening.
    """
    kernel = np.ones((5, 5), np.uint8)

    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    cv2.imwrite(f'./preprocessed/opening_result.tif', opening)
    return opening

def closing(image):
    """
    Applies closing to the input image.

    Parameters:
    image (numpy.ndarray): The input image.

    Returns:
    numpy.ndarray: The image after closing.
    """
    kernel = np.ones((5, 5), np.uint8)
    
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    
    cv2.imwrite(f'./preprocessed/closing_result.tif', closing)
    
    return closing


def transform_hsv(img):
    
    """
    Converts an image from the BGR color space to HSV and saves the HSV channels into TIFF files.

    Parameters:
    img (numpy.ndarray): The input image in the BGR color space.

    Returns:
    tuple: A tuple containing the hue, saturation, and value channels (Hue, Saturation, Value).
    """

    # Convert the image to the HSV color space
    imagem_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Accessing the HSV channels
    matiz = imagem_hsv[:,:,0]
    saturacao = imagem_hsv[:,:,1]
    valor = imagem_hsv[:,:,2]
    cv2.imwrite("./preprocessed/hsv.tif", imagem_hsv)
  
      # Adjusting the channel values for better visualization  
    matiz, saturacao, valor = cv2.split(imagem_hsv)
    matiz = np.uint8(matiz * 2)
    imagem_hsv = np.uint(imagem_hsv * 2)
    saturacao = np.uint8(saturacao * 2)
    
    cv2.imwrite("./preprocessed/matiz.tif", matiz)
    cv2.imwrite("./preprocessed/saturacao.tif", saturacao)
    cv2.imwrite("./preprocessed/valor.tif", valor)

    cv2.imread("./preprocessed/matiz.tif")
    cv2.imread("./preprocessed/saturacao.tif")
    cv2.imread("./preprocessed/valor.tif")
    
    return matiz, saturacao, valor
    