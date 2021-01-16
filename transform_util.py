from torchvision import datasets,transforms

def base_transform(change_size=False, size=224):
    if change_size:
        transform= transforms.Compose([
            transforms.Resize((size,size)),
            transforms.ToTensor(), ])
    else:
        transform = transforms.Compose([
            # transforms.Resize((224,224)),
            transforms.ToTensor(), ])
    return transform

def transform1():
    transform_train = transforms.Compose \
        ([transforms.Resize((32 ,32)),  # resises the image so it can be perfect for our model.
          transforms.RandomHorizontalFlip(),  # FLips the image w.r.t horizontal axis
          transforms.RandomRotation(10),  # Rotates the image to a specified angel
          transforms.RandomAffine(0, shear=10, scale=(0.8 ,1.2)),  # Performs actions like zooms, change shear angles.
          transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Set the color params
          transforms.ToTensor(),  # comvert the image to tensor so that it can work with torch
          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize all the images
          ])


    test_transform = transforms.Compose([transforms.Resize((32 ,32)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ])
    return transform_train, test_transform