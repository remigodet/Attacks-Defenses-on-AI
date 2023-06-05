import os
import pandas as pd
import csv
import numpy as np
import PIL.Image

def read_image_resize(path, height=600, width=600, crop=True, zoom=1):
    img = PIL.Image.open(path)
    if crop:
        x, y = img.size
        img = img.crop((int(x/2-y*zoom/2),int(y/2-y*zoom/2),int(x/2+y*zoom/2),int(y/2+y*zoom/2)))
        # print(img.size)
    img = img.resize((height, width))
    img = np.array(img, dtype=np.uint8)
    return img

def get_img_cropped(image_folder_path, img_path, f=None, max_crop = 0.4, min_crop=1, max_f=250, min_f=55, zoom=None, height=600, width=600):
    '''
    Ex : img = utils.get_img_cropped(image_folder_path, img_path, f=image_focal, max_crop = 0.4, min_crop=1, max_f=250, min_f=55)
    '''
    # print(image_folder_path+"/"+img_path)
    # zoom factor
    if zoom is not None:
        pass
    elif f is not None:
        focal = int(f[1:-2])
        zoom = (min_crop-max_crop)*(focal-min_f) /(max_f-min_f) + max_crop
        # print(focal, zoom)
    else:
        zoom = 1
    # get cropped image
    img = read_image_resize(image_folder_path+"/"+img_path, crop=True, zoom=zoom, height=height, width=width)
    return img

def load_images_df(path):
    jpeg_list = [[f]+os.path.splitext(f)[0].split("_") for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))
                  and f.lower().endswith('.jpg')]
    df = pd.DataFrame(jpeg_list, columns=["filename", "Attaques",	"Attaques2",	"Latitude",	"Longitude",	"distance",	"Focale",	"alpha" ,	"éclairage"])
    return df
def rename_images(folder_path, csv_file):
    # Load the CSV file and extract the names
    names = []
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        print(reader)
        for row in reader:
            names.append(row[0])

    # Get the list of JPEG files in the folder
    jpeg_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))
                  and f.lower().endswith('.jpg')]

    # Check if the number of names matches the number of JPEG files
    if len(names) != len(jpeg_files):
        print("Error: Number of names doesn't match the number of JPEG files.")
        return

    # Rename the JPEG files
    for i, jpeg_file in enumerate(jpeg_files):
        file_name, file_ext = os.path.splitext(jpeg_file)
        new_name = names[i] + file_ext
        new_path = os.path.join(folder_path, new_name)
        old_path = os.path.join(folder_path, jpeg_file)
        os.rename(old_path, new_path)
        print(f"Renamed {jpeg_file} to {new_name}.")

if __name__ == "__main__":
    # Use this to rename images 
    image_folder_path = 'photos/to_rename'
    csv_namefile = 'photos/to_rename/photo_names.csv'
    rename_images(image_folder_path, csv_namefile) # execute once
    print(load_images_df(image_folder_path))
