import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd

store_path = 'path/to/images'  # Replace with the path to your images

# Iterate over images in current directory
counter = 0
df = pd.DataFrame()
brightest_picture = None
darkest_picture = None
all_lines_used_brightest = None
all_lines_used_darkest = None
all_coloums_used_brightest = None
all_coloums_used_darkest = None
for filename in os.listdir('.'):
    if filename.lower().endswith(('.png')):
        img = cv2.imread(filename)
        if img is not None:
            #get infos
            height, width, channels = img.shape
            df = pd.concat([df, pd.DataFrame({'filename': [filename], 'height': [height], 'width': [width], 'channels': [channels]})], ignore_index=True)

        overall_brightness = img.sum()
        # Check for brightest and darkest images
        if brightest_picture is None or overall_brightness > brightest_picture[1]:
            brightest_picture = (img, overall_brightness)
        if darkest_picture is None or overall_brightness < darkest_picture[1]:
            darkest_picture = (img, overall_brightness)

        all_lines_used_brightness = 0
        disqualified = False
        for line in img:
            if line.sum() <= 0:
                disqualified = True
                break
            all_lines_used_brightness += line.sum()
        img_sum = img.sum()
        if not disqualified and img_sum != brightest_picture[0].sum() and img_sum != darkest_picture[0].sum():
            if all_lines_used_brightest is None or all_lines_used_brightness > all_lines_used_brightest[1]:
                all_lines_used_brightest = (img, all_lines_used_brightness)
            if all_lines_used_darkest is None or all_lines_used_brightness < all_lines_used_darkest[1]:
                all_lines_used_darkest = (img, all_lines_used_brightness)
        
        all_coloums_used_brightness = 0
        disqualified = False
        for coloum in img.T:
            if coloum.sum() <= 0:
                disqualified = True
                break
            all_coloums_used_brightness += coloum.sum()
        img_sum = img.sum()
        if not disqualified and img_sum != brightest_picture[0].sum() and img_sum != darkest_picture[0].sum() and img_sum != all_lines_used_brightest[0].sum() and img_sum != all_lines_used_darkest[0].sum():
            if all_coloums_used_brightest is None or all_coloums_used_brightness > all_coloums_used_brightest[1]:
                all_coloums_used_brightest = (img, all_coloums_used_brightness)
            if all_coloums_used_darkest is None or all_coloums_used_brightness < all_coloums_used_darkest[1]:
                all_coloums_used_darkest = (img, all_coloums_used_brightness)

        counter += 1
        if counter % 100 == 0:
            print(f"Processed {counter} images so far...")
print(f"Processed {counter} images.")
#print height
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['height'], label='Height', color='blue')
plt.title('Image Heights')
plt.xlabel('Image Index')
plt.ylabel('Height (pixels)')
plt.savefig(store_path + '/image_heights.png')
plt.clf()
# print width
plt.plot(df.index, df['width'], label='Width', color='green')
plt.title('Image Widths')
plt.xlabel('Image Index')
plt.ylabel('Width (pixels)')
plt.savefig(store_path + '/image_widths.png')
plt.clf()
# print channels
plt.plot(df.index, df['channels'], label='Channels', color='red')
plt.title('Image Channels')
plt.xlabel('Image Index')
plt.ylabel('Channels')
plt.savefig(store_path + 'image_channels.png')
plt.clf()

cv2.imwrite(store_path + 'image1.png', brightest_picture[0])
cv2.imwrite(store_path + 'image2.png', darkest_picture[0])
cv2.imwrite(store_path + 'image3.png', all_lines_used_brightest[0])
cv2.imwrite(store_path + 'image4.png', all_lines_used_darkest[0])
cv2.imwrite(store_path + 'image5.png', all_coloums_used_brightest[0])
cv2.imwrite(store_path + 'all_coloums_used_darkest_image6.png', all_coloums_used_darkest[0])