import os
import ast

import praw
import requests
import re

from PIL import Image

#Read the api info from system enviroment. I have done this as a sucurity feature so that my account is not publicly available
api_info = ast.literal_eval(os.environ.get("REDDIT_API_INFO"))

reddit = praw.Reddit(client_id=api_info["client_id"], \
                     client_secret=api_info["client_secret"], \
                     user_agent=api_info["user_agent"], \
                     username=api_info["username"], \
                     password=api_info["password"])

def crop(image, size=(64, 64)):
    """Crop a PIL.Image to a centered 'size' square
    :param image: a pillow image that will be cropped and resized
    :param size: the size of the output image, defaults to (64, 64)"""
    #find the largest square that fits around the center of the image
    width, height = image.size
    if width > height:
        box = ((width-height)//2, 1, (width+height)//2, height)
    elif width < height:
        box = (1, (height-width)//2, width, (height+width)//2)
    else:
        box = (1, 1, width, height)

    cropped = image.resize(size, box=box, resample=Image.BICUBIC)

    return cropped


class Scraper:
    """A wrapper class for a praw.Subreddit interface with a focus on downloading images"""
    def __init__(self, sub="all"):
        self.sub = reddit.subreddit(sub)

    def query(self, sort_type="top", **kwargs):
        """Given a sort type save the top posts in the subreddit to self.posts

        :param sort_type: the way to sort. Can be "top", "new", "rising", "hot"
        :param kwargs: named arguments used by :class:`praw.subreddit` when getting posts. Ex limit=100"""
        self.posts = getattr(self.sub, sort_type)(**kwargs)

    def save_images(self, path, relative=True):
        """Save the images in self.posts in the dir at the provided path

        :param path: the path to store the images in
        :param relative: whether the path is relative, defaults to True
        :returns: the number of images found and saved"""
        saved = 0
        dir_path = os.path.dirname(os.path.realpath(__file__)) if relative else ""
        for post in self.posts:
            url = (post.url)
            file_name = url.split("/")
            if len(file_name) == 0:
                file_name = re.findall("/(.*?)", url)

            file_name = dir_path+path+file_name[-1]
            if ".jpg" in file_name or ".png" in file_name:
                r = requests.get(url)
                saved += 1
                with open(file_name,"wb") as f:
                    f.write(r.content)

                image = Image.open(file_name)
                image = crop(image)
                # image.show()
                image.save(file_name)

        return saved


if __name__ == '__main__':
    cats = Scraper("cat")
    cats.query("new", limit=50)
    path = "/images/"
    saved = cats.save_images(path)
    print("{} images saved to {}".format(saved, path))