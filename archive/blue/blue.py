from PIL import Image, ImageDraw

im = Image.new("RGB", (64, 64), color="blue")
draw = ImageDraw.Draw(im)
draw.ellipse((13, 13, 50, 50), fill=(255, 255, 0), outline=(255, 0, 0))
for x in range(64):
    im.save(f"{x:02}.png")