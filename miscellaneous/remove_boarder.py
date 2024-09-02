def remove_border_pixels(image, border_size=4):
    # Make a copy of the image to avoid modifying the original
    bordered_image = image.copy()

    # Set top border pixels to black
    bordered_image[:border_size, :] = [0, 0, 0]
    
    # Set bottom border pixels to black
    bordered_image[-border_size:, :] = [0, 0, 0]
    
    # Set left border pixels to black
    bordered_image[:, :border_size] = [0, 0, 0]
    
    # Set right border pixels to black
    bordered_image[:, -border_size:] = [0, 0, 0]
    
    return bordered_image