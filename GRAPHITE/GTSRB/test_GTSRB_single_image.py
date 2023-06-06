import numpy as np
import cv2
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from GTSRBNet import GTSRBNet
from GTSRBDataset import GTSRBDataset
import os
import pandas as pd
import csv
import numpy as np
import PIL.Image

def main(img,label,target):
    # img = argv[1]
    # label = argv[2]
    # target = argv[3]

    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA) # inter area is more stable for downsizing from extremely high res images
    img = img / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1)
    img = transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))(img)
    img_torch = torch.zeros((1, 3, 32, 32))
    img_torch[0, :, :, :] = img

    root = ''


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = GTSRBNet()
    model.to(device)
    
    classes = []
    with open('GTSRB/class_semantics.txt') as f:
        for line in f:
            classes.append(line.strip())

    checkpoint = torch.load('GTSRB/checkpoint_us.tar', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    correct = 0
    total = 0
    tar_top1 = 0
    tar_top2 = 0
    tar_top5 = 0
    lbl_top1 = 0
    lbl_top2 = 0
    lbl_top5 = 0
    with torch.no_grad():
        inputs = img_torch.to(device)
        labels = torch.tensor([int(label)]).to(device)
        labels = labels.long()
        outputs = model(inputs)
        conf, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        conf, lbls = torch.topk(F.softmax(outputs.data, 1), 5, 1)
        if int(label) in lbls[0]:
            lbl_top5 = 1
            if int(label) in lbls[0][:2]:
                lbl_top2 = 1
            if int(label) in lbls[0][0]:
                lbl_top1 = 1
        if int(target) in lbls[0]:
            tar_top5 = 1
            if int(target) in lbls[0][:2]:
                tar_top2 = 1
            if int(target) in lbls[0][0]:
                tar_top1 = 1
        print("Predicted class: ", classes[predicted[0].item()], " Predicted confidence: ", F.softmax(outputs.data, 1)[0][predicted[0].item()].item(), " Target confidence: ", F.softmax(outputs.data, 1)[0][int(target)].item(), " Stop confidence: ", F.softmax(outputs.data, 1)[0][int(label)].item())
        print(predicted[0].item(),int(target))
        print("topk:", tar_top1, tar_top2, tar_top5, lbl_top1, lbl_top2, lbl_top5)
        print("conf:", F.softmax(outputs.data, 1)[0][int(target)].item(), F.softmax(outputs.data, 1)[0][int(label)].item())

    val_acc = 100.0 * correct / total
    print('Val accuracy: %.3f' % (val_acc))
    return classes[predicted[0].item()]

def main_noprint(img,label,target):
    # img = argv[1]
    # label = argv[2]
    # target = argv[3]

    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA) # inter area is more stable for downsizing from extremely high res images
    img = img / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1)
    img = transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))(img)
    img_torch = torch.zeros((1, 3, 32, 32))
    img_torch[0, :, :, :] = img

    root = ''


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = GTSRBNet()
    model.to(device)
    
    classes = []
    with open('GTSRB/class_semantics.txt') as f:
        for line in f:
            classes.append(line.strip())

    checkpoint = torch.load('GTSRB/checkpoint_us.tar', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    correct = 0
    total = 0
    tar_top1 = 0
    tar_top2 = 0
    tar_top5 = 0
    lbl_top1 = 0
    lbl_top2 = 0
    lbl_top5 = 0
    with torch.no_grad():
        inputs = img_torch.to(device)
        labels = torch.tensor([int(label)]).to(device)
        labels = labels.long()
        outputs = model(inputs)
        conf, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        conf, lbls = torch.topk(F.softmax(outputs.data, 1), 5, 1)
        if int(label) in lbls[0]:
            lbl_top5 = 1
            if int(label) in lbls[0][:2]:
                lbl_top2 = 1
            if int(label) in lbls[0][0]:
                lbl_top1 = 1
        if int(target) in lbls[0]:
            tar_top5 = 1
            if int(target) in lbls[0][:2]:
                tar_top2 = 1
            if int(target) in lbls[0][0]:
                tar_top1 = 1
    val_acc = 100.0 * correct / total
    return classes[predicted[0].item()]

def read_image_resize(path, height=600, width=600, crop=True, zoom=1):
    img = cv2.imread(path)
    if crop:
        print("Size before crop : " +str(img.shape))
        x, y, z= img.shape
        img = img[int(x/2-x*zoom/2):int(x/2+x*zoom/2),int(y/2-y*zoom/2):int(y/2+y*zoom/2)]
        print("Size after crop : " +str(img.shape))
    img = np.array(img)
    return img

def get_img_cropped(image_folder_path, img_path, f=None, max_crop = 0.4, min_crop=1, max_f=250, min_f=55 ):
    '''
    Ex : img = utils.get_img_cropped(image_folder_path, img_path, f=image_focal, max_crop = 0.4, min_crop=1, max_f=250, min_f=55)
    '''
    # print(image_folder_path+"/"+img_path)
    # zoom factor
    if f is not None:
        focal = int(f[1:-2])
        zoom = (min_crop-max_crop)*(focal-min_f) /(max_f-min_f) + max_crop
        print(focal, zoom)
    else:
        zoom = 1
    # get cropped image
    img = read_image_resize(image_folder_path+"/"+img_path, crop=True, zoom=zoom)
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

    ## MANUAL CROP

    L = os.listdir('images_cropped_manually')
    temoin_success = 0
    temoin_total = 0
    attack_success=0
    attack_total=0
    for path in L :
        result = main_noprint("images_cropped_manually/"+path,14,2)
        print(path + "    " +result)
        if path[0:6] == "témoin" :
            temoin_total +=1
            if result == "Stop":
                temoin_success +=1
        if path[0:13] == "graphite_stop" :
            attack_total +=1
            if result == "Speed limit (30km/h)":
                attack_success +=1
    print("------------------------ GLOBAL RESULTS --------------------------")
    print("Success rate on witness :" +str(temoin_success) + "/" + str(temoin_total) + "=" + str(temoin_success/temoin_total))
    print("Success rate on attacks :" +str(attack_success) + "/" + str(attack_total) + "=" + str(attack_success/attack_total))

    
    ## AUTOMATIC CROP

    # # Use this to rename images 
    # image_folder_path = 'images_eval'
    # csv_namefile = 'names_photos.csv'
    # # rename_images(image_folder_path, csv_namefile) # execute once
    # df = load_images_df(image_folder_path)
    # print(df.head())
    # col_filename = df.loc[:,"filename"]
    # col_focale = df.loc[:,"Focale"]
    # for i in range(29):
    #     img_path = "images_eval/" +col_filename[i]
    #     img_tested_path = "images_tested/"+col_filename[i]
    #     image_focal = col_focale[i]
    #     img = get_img_cropped(image_folder_path, col_filename[i], f=image_focal, max_crop = 0.4, min_crop=1, max_f=250, min_f=55)
    #     os.chdir('images_tested')
    #     cv2.imwrite(col_filename[i],img)
    #     os.chdir('../')
    #     print("Image saved to : images_tested/"+col_filename[i], )
    #     label = 2
    #     target = 14
    #     main(img_tested_path,label,target)
    #     print("----------------------NEXT IMAGE---------------------")


