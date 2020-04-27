import os


def load_paths(dir):
    """
    Get the list of image paths in a certain directory.
    """
    IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                      '.png', '.PNG', '.bmp', '.BMP']
    image_paths = []

    # traverse directory to obtain only paths to images
    for dir_name, _, paths in sorted(os.walk(os.path.expand_user(dir))):
        for path in paths:
            if any(path.endswith(extensions) for extensions in IMG_EXTENSIONS):
                image_paths.append(os.path.expanduser(dir_name + '/' + path))

    return image_paths


def save_image(image, image_path):
    """
    Save an image to disk.
    """
    image = ((image[0] + 1) * 127.5).astype(np.uint8) # convert from [-1, 1] to [0, 255]
    img = Image.fromarray(image)
    img.save(os.path.expanduser(image_path))
