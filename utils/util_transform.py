from torchvision import datasets,transforms

def base_transform(args):
    # cifar size (32,32)
    # ImageNet(224,224)

    transform= transforms.Compose([
        transforms.Resize((args.img_size,args.img_size)),
        transforms.ToTensor(), ])

    return transform

def normalize_transform(args):
    # cifar size (32,32)
    # ImageNet(224,224)

    transform= transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

    return transform

def transform(args):
    train_normalization = {mean:(0.5, 0.5, 0.5),
                     var: (0.5, 0.5, 0.5)}
    test_normalization = {mean: (0.5, 0.5, 0.5),
                           var: (0.5, 0.5, 0.5)}

    transform_train = transforms.Compose \
        ([transforms.Resize((args.img_size ,args.img_size)),  # resises the image so it can be perfect for our model.
          transforms.RandomHorizontalFlip(),  # FLips the image w.r.t horizontal axis
          transforms.RandomRotation(10),  # Rotates the image to a specified angel
          transforms.RandomAffine(0, shear=10, scale=(0.8 ,1.2)),  # Performs actions like zooms, change shear angles.
          transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Set the color params
          transforms.ToTensor(),  # comvert the image to tensor so that it can work with torch
          transforms.Normalize(train_normalization['mean'], train_normalization['var'])  # Normalize all the images
          ])


    test_transform = transforms.Compose([transforms.Resize((args.img_size ,args.img_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(test_normalization['mean'], test_normalization['var'])
                                    ])

    return transform_train, test_transform

