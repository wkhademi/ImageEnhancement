import os
import sys
import cv2


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


def video2photos(video_path, save_dir):
    """
    Convert a video to a set of images and save them.
    """
    count = 0
    video_name = video_path.split('/')[-1][:-4]

    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()

    while success:
        cv2.imwrite('%s%s_frame%d.jpg'%(save_dir, video_name, count), image)
        success, image = vidcap.read()
        count += 1


def load_video_paths(dir):
    """
    Get the list of video paths in a certain directory.
    """
    VIDEO_EXTENSIONS = ['.mov', '.MOV', '.mp4']
    video_paths = []

    # traverse directory to obtain only paths to videos
    for dir_name, _, paths in sorted(os.walk(os.path.expanduser(dir))):
        for path in paths:
            if any(path.endswith(extensions) for extensions in VIDEO_EXTENSIONS):
                video_paths.append(os.path.expanduser(dir_name + '/' + path))

    return video_paths
