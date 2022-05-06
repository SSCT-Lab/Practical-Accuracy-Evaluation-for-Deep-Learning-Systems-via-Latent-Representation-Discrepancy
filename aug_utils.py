from torchvision import transforms

cifar_good_trans = transforms.Compose([
    transforms.RandomRotation(degrees=(-10, 10)),
    transforms.ColorJitter(brightness=(0.8, 1.3), contrast=(0.8, 1.3)),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

cifar_aug_trans = transforms.Compose([
    transforms.RandomRotation(degrees=(-10, 10)),
    transforms.ColorJitter(brightness=(0.8, 1.3), contrast=(0.8, 1.3)),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

svhn_good_trans = transforms.Compose([
    transforms.RandomRotation(degrees=(-20, 20)),
    transforms.ColorJitter(brightness=(0.7, 1.5), contrast=(0.7, 1.5)),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.GaussianBlur(5, 5),
    transforms.RandomAffine(0, None, (0.8, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614)),
])

svhn_aug_trans = svhn_good_trans

mnist_good_trans = transforms.Compose([
    transforms.RandomRotation(degrees=(-20, 20)),
    transforms.ColorJitter(brightness=(0.8, 1.5), contrast=(0.8, 1.5)),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.GaussianBlur(5, 5),
    transforms.RandomAffine(0, None, (0.7, 1.2)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,)),
])

mnist_aug_trans = mnist_good_trans

fashion_good_trans = transforms.Compose([
    transforms.RandomRotation(degrees=(-10, 10)),
    transforms.ColorJitter(brightness=(0.7, 1.3), contrast=(0.7, 1.3)),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    # transforms.GaussianBlur(3, 5),
    transforms.RandomAffine(0, None, (0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,)),
])

fashion_aug_trans = fashion_good_trans

cifar_bad_trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomChoice(
        [transforms.RandomErasing(p=0.8, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=(254 / 255, 0, 0)),
         transforms.RandomErasing(p=0.8, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random')]),
    transforms.RandomChoice(
        [transforms.ColorJitter(brightness=(0.1, 0.5), contrast=(0.1, 0.5), saturation=(0.1, 0.5)),
         transforms.ColorJitter(brightness=(5, 10), contrast=(5, 10), saturation=(5, 10))]),
])

svhn_bad_trans = cifar_bad_trans

mnist_bad_trans = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.RandomChoice(
        [transforms.RandomErasing(p=0.8, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=(254 / 255, 0, 0)),
         transforms.RandomErasing(p=0.8, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random')]),
    transforms.RandomChoice(
        [transforms.ColorJitter(brightness=(0.1, 0.5), contrast=(0.1, 0.5), saturation=(0.1, 0.5)),
         transforms.ColorJitter(brightness=(5, 10), contrast=(5, 10), saturation=(5, 10))]),
    transforms.Grayscale(num_output_channels=1),
])

fashion_bad_trans = mnist_bad_trans

cifar_train_trans = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

svhn_train_trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614)),
])

mnist_train_trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

fashion_train_trans = mnist_train_trans

cifar_test_trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

svhn_test_trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614)),
])

mnist_test_trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

fashion_test_trans = mnist_test_trans

### For RQ3_gen
cifar_small_trans = transforms.Compose([
    transforms.RandomRotation(degrees=(-2, 2)),
    transforms.ColorJitter(brightness=(0.95, 1.05), contrast=(0.95, 1.05)),
    transforms.RandomAffine(degrees=0, translate=(0.01, 0.01)),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

cifar_large_trans = transforms.Compose([
    transforms.RandomRotation(degrees=(-10, 10)),
    transforms.ColorJitter(brightness=(0.8, 1.3), contrast=(0.8, 1.3)),
    transforms.GaussianBlur(3, 5),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

svhn_small_trans = transforms.Compose([
    transforms.RandomRotation(degrees=(-5, 5)),
    transforms.ColorJitter(brightness=(0.95, 1.05), contrast=(0.95, 1.05)),
    transforms.RandomAffine(degrees=0, translate=(0.01, 0.01)),
    # transforms.GaussianBlur(5, 5),
    transforms.RandomAffine(0, None, (0.9, 1.05)),
    transforms.ToTensor(),
    transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614)),

])

svhn_large_trans = transforms.Compose([
    transforms.RandomRotation(degrees=(-30, 30)),
    transforms.ColorJitter(brightness=(0.7, 1.5), contrast=(0.7, 1.5)),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.GaussianBlur(5, 5),
    transforms.RandomAffine(0, None, (0.8, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614)),

])

mnist_small_trans = transforms.Compose([
    transforms.RandomRotation(degrees=(-5, 5)),
    transforms.ColorJitter(brightness=(0.95, 1.05), contrast=(0.95, 1.05)),
    transforms.RandomAffine(degrees=0, translate=(0.01, 0.01)),
    # transforms.GaussianBlur(5, 5),
    transforms.RandomAffine(0, None, (0.95, 1.05)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,)),
])

mnist_large_trans = transforms.Compose([
    transforms.RandomRotation(degrees=(-35, 35)),
    transforms.ColorJitter(brightness=(0.7, 1.6), contrast=(0.7, 1.6)),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.GaussianBlur(5, 5),
    transforms.RandomAffine(0, None, (0.6, 1.3)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,)),
])

fashion_small_trans = transforms.Compose([
    transforms.RandomRotation(degrees=(-3, 3)),
    transforms.ColorJitter(brightness=(0.95, 1.05), contrast=(0.95, 1.05)),
    transforms.RandomAffine(degrees=0, translate=(0.01, 0.01)),
    # transforms.GaussianBlur(3, 5),
    # transforms.RandomAffine(0, None, (0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,)),
])

fashion_large_trans = transforms.Compose([
    transforms.RandomRotation(degrees=(-30, 30)),
    transforms.ColorJitter(brightness=(0.6, 1.4), contrast=(0.6, 1.4)),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.GaussianBlur(3, 5),
    transforms.RandomAffine(0, None, (0.8, 1.2)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,)),
])